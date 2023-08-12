import errno
import queue
import select
import socket
import struct
import threading
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np


def remote_connection_closed(sock: socket.socket) -> bool:
    """
    Returns True if the remote side did close the connection

    """
    try:
        buf = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        if buf == b'':
            return True
    except BlockingIOError as exc:
        if exc.errno != errno.EAGAIN:
            # Raise on unknown exception
            raise
    except OSError:
        return True
    return False


class PlaybackState (Enum):
    PLAYING = 0
    PAUSED = 1
    FAST_FORWARD = 2


class RecvMessageType (Enum):
    ACTIVE_FILES = 0
    STATE_INACTIVE = 1
    PLAYBACK_RATE = 2
    STATE_PLAYING = 3
    STATE_PAUSED = 4
    STATE_FAST_FORWARD = 5
    CURRENT_TICK = 6
    DEBUG_TICK = 7
    PROCESSED_SCRIPT = 10
    ENTITY_INFO = 100
    GAME_LOCATION = 255


class EntityInfo:
    state: bool

    x: float
    y: float
    z: float

    pitch: float
    yaw: float
    roll: float

    vx: float
    vy: float
    vz: float

    def dist_sq(self, x=None, y=None, z=None, yaw=None, pitch=None, roll=None, vx=None, vy=None, vz=None) -> float:
        """
        Compute the distance squared to the given coords.
        Be careful not to compare values for which you have given
        different input counts/types!
        """
        dist = 0

        if x is not None:
            dist += (self.x - x)**2
        if y is not None:
            dist += (self.y - y)**2
        if z is not None:
            dist += (self.z - z)**2
        if pitch is not None:
            dist += (self.pitch - pitch)**2
        if yaw is not None:
            dist += (self.yaw - yaw)**2
        if roll is not None:
            dist += (self.roll - roll)**2
        if vx is not None:
            dist += (self.vx - vx)**2
        if vy is not None:
            dist += (self.vy - vy)**2
        if vz is not None:
            dist += (self.vz - vz)**2

        return dist

    def __str__(self) -> str:
        res = "EntityInfo: \n"
        if self.state:
            res += f"\tpos: [{self.x}, {self.y}, {self.z}]\n"
            res += f"\tang: [{self.pitch}, {self.yaw}, {self.roll}]\n"
            res += f"\tvel: [{self.vx}, {self.vy}, {self.vz}]\n"
        else:
            res += "\tEntity not found\n"
        return res


class GameInstanceBase:
    """
    Generic building block for the two possible communication scenarios
    SAR server, this client or
    SAR client, this server
    """

    def __init__(self):
        # The socket field should be initialized by the parent class
        self.sock: socket.socket

        # Server state
        self.active_scripts: List[str] = []
        self.state: PlaybackState = PlaybackState.PLAYING
        self.playback_rate: float = 1.0
        self.current_tick: int = 0
        self.debug_tick: int = 0
        self.game_location = ""

    def set_sock(self, sock: socket.socket):
        """
        Sets this server to communicate with the given socket
        The purpose of this 
        """
        self.sock = sock

    # =============================================================
    #                             Send
    # =============================================================

    def start_file_playback(self, script_path1: str, script_path2: str = ""):
        """
        Request a playback of the given file(s)
        """
        packet = b''
        packet += struct.pack("!B", 0)
        packet += struct.pack("!I", len(script_path1))
        packet += script_path1.encode()
        packet += struct.pack("!I", len(script_path2))
        packet += script_path2.encode()

        self.sock.send(packet)
        self.current_tick = 0

    def start_content_playback(self, script1: str, script2: str = ""):
        """
        Request a playback of the given scripts
        """
        script1_name = "script1"
        script2_name = "script2" if script2 != "" else ""
        packet = b''
        packet += struct.pack("!B", 10)
        for string in [script1_name, script1, script2_name, script2]:
            packet += struct.pack("!I", len(string))
            packet += string.encode()

        self.sock.send(packet)

    def stop_playback(self):
        """
        Request for the playback to stop
        """
        self.sock.send(struct.pack("!B", 1))

    def change_playback_speed(self, speed: float):
        """
        Request a change to the playback speed
        """
        packet = b''
        packet += struct.pack("!B", 2)
        packet += struct.pack("!f", speed)

        self.sock.send(packet)

    def resume_playback(self):
        """
        Request for the playback to resume
        """
        self.sock.send(struct.pack("!B", 3))

    def pause_playback(self):
        """
        Request for the playback to pause
        """
        self.sock.send(struct.pack("!B", 4))

    def fast_forward(self, to_tick=0, pause_after=True):
        """
        Request fast-forward
        """
        packet = b''
        packet += struct.pack("!B", 5)
        packet += struct.pack("!I", to_tick)
        packet += struct.pack("!?", pause_after)

        self.sock.send(packet)

    def pause_at(self, tick=0):
        """
        Request for the playback to pause at the given tick
        """
        packet = b''
        packet += struct.pack("!B", 6)
        packet += struct.pack("!I", tick)

        self.sock.send(packet)

    def advance_playback(self):
        """
        Request for the playback to advance a single tick
        """
        self.sock.send(struct.pack("!B", 7))

    def entity_info(self, entity_selector="player"):
        """
        Request information on an entity, player is default
        """
        packet = b''
        packet += struct.pack("!B", 100)
        packet += struct.pack("!I", len(entity_selector))
        packet += entity_selector.encode()

        self.sock.send(packet)

    def entity_info_continuous(self, entity_selector="player"):
        """
        Request continuous information on an entity, player is default
        """
        packet = b''
        packet += struct.pack("!B", 101)
        packet += struct.pack("!I", len(entity_selector))
        packet += entity_selector.encode()

        self.sock.send(packet)

    def entity_info_continuous_stop(self):
        """
        Stop recieving contiuous entity info
        """
        self.entity_info_continuous("")

    # =============================================================
    #                            Receive
    # =============================================================

    def recieve(self) -> List[EntityInfo]:
        """
        Recieve all pending data from the server. Non blocking.
        """
        entity_info_list = []

        # readable will contain our socket only if there is data to read
        readable, w, e = select.select([self.sock], [], [], 0)

        while len(readable) > 0:
            # read data
            _, ent_info = self.__recv_blocking()
            if ent_info is not None:
                entity_info_list.append(ent_info)
            # Check if there is more data
            readable, w, e = select.select([self.sock], [], [], 0)

        return entity_info_list

    def recieve_until(self, message_type=RecvMessageType.CURRENT_TICK) -> List[EntityInfo]:
        """
        Recieve all pending data and return once a specific packet type is recieved. Data may still be left unread.
        """
        entity_info_list = []

        while True:
            # read data
            msg_type, ent_info = self.__recv_blocking()

            if ent_info is not None:
                entity_info_list.append(ent_info)

            if msg_type == message_type:
                return entity_info_list

    def __recv_blocking(self):
        ent_info = None
        msg_type = self.__recv(1)
        try:
            msg_type = RecvMessageType(struct.unpack("!B", msg_type)[0])
        except:
            self.sock.close()
            raise
        
        # print(msg_type)

        if msg_type == RecvMessageType.ACTIVE_FILES:
            len1 = struct.unpack("!I", self.__recv(4))[0]
            if len1 > 0:
                self.active_scripts.append(self.__recv(len1).decode())
            len2 = struct.unpack("!I", self.__recv(4))[0]
            if len2 > 0:
                self.active_scripts.append(self.__recv(len2).decode())
        elif msg_type == RecvMessageType.STATE_INACTIVE:
            self.active_scripts.clear()
        elif msg_type == RecvMessageType.PLAYBACK_RATE:
            self.playback_rate = struct.unpack("!f", self.__recv(4))[0]
        elif msg_type == RecvMessageType.STATE_PLAYING:
            self.state = PlaybackState.PLAYING
        elif msg_type == RecvMessageType.STATE_PAUSED:
            self.state = PlaybackState.PAUSED
        elif msg_type == RecvMessageType.STATE_FAST_FORWARD:
            self.state = PlaybackState.FAST_FORWARD
        elif msg_type == RecvMessageType.CURRENT_TICK:
            self.current_tick = struct.unpack("!I", self.__recv(4))[0]
        elif msg_type == RecvMessageType.DEBUG_TICK:
            self.debug_tick = struct.unpack("!i", self.__recv(4))[0]
        elif msg_type == RecvMessageType.PROCESSED_SCRIPT:
            self.active_scripts.clear()
            slot = struct.unpack("!B", self.__recv(1))[0]
            raw_script_len = struct.unpack("!I", self.__recv(4))[0]
            raw_script = self.__recv(raw_script_len)
        elif msg_type == RecvMessageType.ENTITY_INFO:
            ent_info = EntityInfo()
            info_state = struct.unpack("!B", self.__recv(1))[0]
            ent_info.state = info_state != 0

            if ent_info.state:  # The rest of the data is present
                ent_data = struct.unpack("!fffffffff", self.__recv(4*9))
                ent_info.x = ent_data[0]
                ent_info.y = ent_data[1]
                ent_info.z = ent_data[2]
                ent_info.pitch = ent_data[3]
                ent_info.yaw = ent_data[4]
                ent_info.roll = ent_data[5]
                ent_info.vx = ent_data[6]
                ent_info.vy = ent_data[7]
                ent_info.vz = ent_data[8]

        elif msg_type == RecvMessageType.GAME_LOCATION:
            str_len = struct.unpack("!I", self.__recv(4))[0]
            if str_len > 0:
                self.game_location = self.__recv(str_len).decode()

        return (msg_type, ent_info)

    def __recv(self, size: int) -> bytes:
        """
        Recieve size bytes from the socket and return them. Times out after 25s.
        """
        before = time.time()
        after = time.time()

        buf = b''
        while len(buf) < size:
            buf += self.sock.recv(size - len(buf))
            after = time.time()

            if (after - before) > 25:
                raise

        return buf

    def __str__(self) -> str:
        res = "TasServer: \n"
        res += f"\tactive_scripts: {str(self.active_scripts)}\n"
        res += f"\tstate: {str(self.state)}\n"
        res += f"\tplayback_rate: {str(self.playback_rate)}\n"
        res += f"\tcurrent_tick: {str(self.current_tick)}\n"
        res += f"\tdebug_tick: {str(self.debug_tick)}\n"
        res += f"\tgame_location: {str(self.game_location)}\n"
        return res


class TasServer(GameInstanceBase):
    def __init__(self, ip="127.0.0.1", port=6555):
        super().__init__()

        self.ip = ip
        self.port = port

    def connect(self):
        """
        Initiate a connection to the server
        """
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.connect((self.ip, self.port))


class TasClient(GameInstanceBase):
    def __init__(self, sock):
        super().__init__()
        self.sock = sock


MeasureFn = Callable[[GameInstanceBase, List[float]], float]


class ServerPool:
    # bool true if the server is available
    # servers: Dict[TasServer, bool]

    def __init__(self, measure: MeasureFn, addresses=[("127.0.0.1", 6555)], init: Union[str, None] = None) -> None:
        self.init_script = init
        self.measure = measure
        self.addresses = addresses

    def process_brute(self, points: List[List[float]]) -> Any:
        results = [float('inf')] * len(points)

        # Create the point queue
        q = queue.Queue()
        for point in enumerate(points):
            q.put(point)

        # Check all threads are alive. If not, try to start it again
        threads: Dict[Tuple[str, int], ServerHandlerThread] = {}
        while not q.empty():
            for address in self.addresses:
                thread = threads.get(address)
                if thread is None or not thread.is_alive():
                    try:
                        threads[address] = ServerHandlerThread(
                            self.init_script, self.measure, address, q, results)
                        threads[address].start()
                    except:
                        pass
            time.sleep(10)

        q.join()

        return np.array(results)


class ServerHandlerThread(threading.Thread):
    _target: MeasureFn

    def __init__(self, init_script: Union[str, None], target: MeasureFn, address: Tuple[str, int], queue: queue.Queue, results) -> None:
        super().__init__(None, target, None, [], None, daemon=True)
        self.q = queue
        self.results = results
        self.init_script = init_script
        self.game = TasServer(address[0], address[1])
        self.game.connect()

    def run(self):
        try:
            if self.init_script is not None:
                self.game.recieve()
                if len(self.game.active_scripts) > 0:
                    self.game.stop_playback()
                    self.game.recieve_until(RecvMessageType.PROCESSED_SCRIPT)
                self.game.fast_forward()  # reset ff
                self.game.start_content_playback(self.init_script)
                self.game.recieve_until(RecvMessageType.PROCESSED_SCRIPT)
        except:
            return

        # Run tasks, until the queue is empty
        while True:
            # Get the next task
            try:
                i, point = self.q.get(False)
            except:
                return

            # Try to compute the point
            try:
                self.results[i] = self._target(self.game, point)
            except Exception as e:
                # In case of failure, put the task back in the queue and exit
                print(e)
                # print(traceback.format_exc())
                self.q.put((i, point))
                self.q.task_done()
                return

            self.q.task_done()


class ClientPool:
    def __init__(self, measure: MeasureFn, init: Union[str, None] = None) -> None:
        self.init_script = init
        self.measure = measure

        # List of client threads
        self.clients: Dict[socket._RetAddress, socket.socket] = {}

    def listen(self, port: int):
        self.sock = socket.socket(socket.AF_INET)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", port))
        self.sock.listen(5)

        # Start thread to accept connections
        threading.Thread(target=self.handle_connections, daemon=True).start()

    def handle_connections(self):
        while True:  # TODO: better condition, maybe cap the number of clients?
            conn, addr = self.sock.accept()
            self.clients[addr] = conn
            conn.settimeout(20)
            print(f"Client connected. Total clients: {len(self.clients)}")

    def process_brute(self, points: List[List[float]]) -> Any:
        results = [float('inf')] * len(points)
        # Create the point queue
        q = queue.Queue()
        for point in enumerate(points):
            q.put(point)

        # Check all threads are alive. If not, try to start it again
        threads: Dict[socket.socket, ClientHandlerThread] = {}
        while q.unfinished_tasks > 0:
            to_delete = []
            to_start = []

            items = list(self.clients.items())
            for addr, conn in items:
                if remote_connection_closed(conn):
                    to_delete.append(addr)
                    continue
                thread = threads.get(conn)
                if thread is None or not thread.is_alive():
                    to_start.append((addr, conn))
            
            for addr, conn in to_start:
                try:
                    threads[conn] = ClientHandlerThread(
                        self.init_script, self.measure, conn, q, results)
                    threads[conn].start()
                except:
                    print("Client thread failed to start! Dropping.")
                    to_delete.append(addr)
            
            for addr in to_delete:
                del self.clients[addr]
                print(f"Client disconnected. {len(self.clients)} clients left.")
            
            time.sleep(10)

        q.join()

        return np.array(results)
    pass


class ClientHandlerThread(threading.Thread):
    _target: MeasureFn

    def __init__(self, init_script: Union[str, None], target: MeasureFn, sock: socket.socket, queue: queue.Queue, results) -> None:
        super().__init__(None, target, None, [], None, daemon=True)
        self.q = queue
        self.results = results
        self.init_script = init_script
        self.game = TasClient(sock)

    def run(self):
        try:
            if self.init_script is not None:
                self.game.recieve()
                self.game.stop_playback()
                self.game.fast_forward()  # reset ff
                time.sleep(0.1)
                self.game.recieve()
                self.game.start_content_playback(self.init_script)
                self.game.recieve_until(RecvMessageType.PROCESSED_SCRIPT)
        except:
            print("Client failed to run init script! Retrying.")
            return

        # Run tasks, until the queue is empty
        while True:
            # Get the next task
            try:
                i, point = self.q.get(False)
            except:
                return

            # Try to compute the point
            try:
                self.results[i] = self._target(self.game, point)
            except Exception as e:
                # In case of failure, put the task back in the queue and exit
                print(e)
                # print(traceback.format_exc())
                self.q.put((i, point))
                self.q.task_done()
                return

            self.q.task_done()
