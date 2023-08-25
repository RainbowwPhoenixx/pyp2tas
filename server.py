import datetime
import errno
import logging
import queue
import select
import socket
import struct
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib
import matplotlib.ticker as plt_ticker
import numpy as np
import psutil
from matplotlib import pyplot as plt
from sko.PSO import PSO
from sko.tools import set_run_mode
import parallel_coord_plot as pcplot

from denylist import DENY_LIST

matplotlib.use("Qt5agg")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    MESSAGE = 8
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
        
        self.messages: List[str] = []

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

    def readable(self) -> bool:
        readable, w, e = select.select([self.sock], [], [], 0)
        return len(readable) > 0

    def recieve(self) -> List[EntityInfo]:
        """
        Recieve all pending data from the server. Non blocking.
        """
        entity_info_list = []

        while self.readable():
            # read data
            _, ent_info = self.__recv_blocking()
            if ent_info is not None:
                entity_info_list.append(ent_info)

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
        msg_type = RecvMessageType(struct.unpack("!B", msg_type)[0])

        logging.debug(msg_type)

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
        elif msg_type == RecvMessageType.MESSAGE:
            message_len = struct.unpack("!i", self.__recv(4))[0]
            message = self.__recv(message_len).decode()
            self.messages.append(message)
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

    def __recv(self, size: int, timeout: float = 30) -> bytes:
        """
        Recieve size bytes from the socket and return them.
        """
        before = time.time()
        after = time.time()

        buf = b''
        while len(buf) < size:
            if self.readable():
                buf += self.sock.recv(size - len(buf))
            else:
                if (after - before) > 5:
                    logging.warn(f"({self.sock.getpeername()}) client is being slow!")
                time.sleep(.5)
            after = time.time()

            if (after - before) > timeout:
                raise socket.timeout

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
NewBestAction = Callable[[str, List[float], float], None]

class ClientPool:
    def __init__(self, measure: MeasureFn, init: Union[str, None] = None, on_pb: Union[NewBestAction, None] = None) -> None:
        self.init_script = init
        self.measure = measure
        self.on_pb_action = on_pb
        self.pso = None

        # List of client threads
        self.clients: Dict[socket._RetAddress, ClientHandlerThread] = {}

        # Pending tasks
        self.tasks_q = queue.Queue()
        self.results_q = queue.Queue()

        # Various stats
        self.client_data_tracker = ClientDataTracker()
        self.pworst_hist: List[float] = []
        self.pavg_hist: List[float] = []
        self.pbest_diff_hist: List[float] = []
        self.gbest_x_hist: List[List[float]] = []
        self.iter_end_times: List[float] = []

        # System stats
        self.cpu_percent_hist: List[Tuple[float, float]] = []

    def listen(self, port: int):
        self.sock = socket.socket(socket.AF_INET)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", port))
        self.sock.listen(10)

        # Start thread to accept connections
        self.accept_thread = threading.Thread(target=self.handle_connections, daemon=True).start()

    def handle_connections(self):
        while True:
            try:
                while True:  # TODO: better condition, maybe cap the number of clients?
                    conn, addr = self.sock.accept()
                    if addr[0] in DENY_LIST:
                        conn.close()
                        continue
                    client = ClientHandlerThread(self.init_script, self.measure, conn, self.tasks_q, self.results_q)
                    client.start()
                    self.clients[addr] = client
                    conn.settimeout(60)
                    logging.info(f"Client connected. Total clients: {len(self.clients)}")
            except:
                logging.exception("The connection accept thread crashed. Restarting.")

    def process_brute(self, points: List[List[float]]) -> Any:
        proc = psutil.Process()
        results = [float('inf')] * len(points)

        # Create the point queue
        for point in enumerate(points):
            self.tasks_q.put(point)

        if self.pso is not None:
            self.pbest_diff_hist.append(np.amax(self.pso.pbest_y) - np.amin(self.pso.pbest_y))
            self.pavg_hist.append(float(np.mean(self.pso.pbest_y)))
            self.pworst_hist.append(np.amax(self.pso.pbest_y))

        # Check all threads are alive. If not, try to start it again
        while self.tasks_q.unfinished_tasks > 0:
            items = list(self.clients.items())
            for addr, thread in items:
                self.client_data_tracker.set_client_name(addr[0], thread.contributor)
                if not thread.is_alive():
                    del self.clients[addr]
                    thread.close()
                    logging.info(f"Client disconnected. {len(self.clients)} clients left.")

            time_elapsed = (time.time() - self.start_time)/60
            if self.client_data_tracker.get_total_clients() != len(self.clients):
                items = list(self.clients.items())
                counts = {}
                for addr, thread in items:
                    counts[addr[0]] = 1 + counts.get(addr[0], 0)
                
                self.client_data_tracker.set_client_count(counts, time_elapsed)
            
            cpu_pct = proc.cpu_percent()
            if cpu_pct > 1:
                self.cpu_percent_hist.append((time_elapsed, cpu_pct))
            
            self.make_plots()
            time.sleep(5)
        
        # Collect results
        while not self.results_q.empty():
            idx, value = self.results_q.get()
            results[idx] = value
        
        # Collect stats
        if self.pso is not None:
            best_idx = int(np.argmin(results))
            if self.pso.gbest_y > results[best_idx]:
                self.gbest_x_hist.append(points[best_idx])
                if self.on_pb_action is not None:
                    self.on_pb_action(datetime.datetime.fromtimestamp(self.start_time).isoformat(), points[best_idx], results[best_idx])

        self.iter_end_times.append(time.time())

        self.tasks_q.join()
        self.iter += 1

        return np.array(results)
    
    def pso_optimise(self, bounds: List[List[float]], pop: int, iters: int):
        self.start_time = time.time()
        self.iter = 0

        set_run_mode(self.process_brute, "vectorization")

        self.pso = PSO(
            func=self.process_brute,
            n_dim=len(bounds[0]),
            pop=pop,
            max_iter=iters-1,
            lb=bounds[0],  # type: ignore
            ub=bounds[1],  # type: ignore
            w=0.8, c1=0.5, c2=0.5
        )
        self.pso.lb = -np.array([np.inf] * len(bounds[0]))
        self.pso.ub = np.array([np.inf] * len(bounds[0]))
        try:
            self.pso.run()
        except Exception as e:
            logging.exception('')
            logging.error("A problem occurred while optimising. Returning early.")

        return self.pso.gbest_x, self.pso.gbest_y
        
    def make_plots(self):
        plt.ion()
        # plt.style.use('dark_background')
        # Set background color of the outer
        # area of the plt
        plt.figure(facecolor='#2e3035')

        elapsed = time.time() - self.start_time
        rows = 2
        colums = 2

        cost_graph = plt.subplot2grid((rows, colums), (0, 0))
        time_per_iter_graph = plt.subplot2grid((rows, colums), (1, 0))
        clients_graph = plt.subplot2grid((rows, colums), (0, 1))
        cpu_graph = plt.subplot2grid((rows, colums), (1, 1))
        # coords_graph = plt.subplot2grid((rows, colums), (1, 1), colspan=2)
        all_graphs = [cost_graph, time_per_iter_graph, clients_graph, cpu_graph]
        for graph in all_graphs:
            graph.set_facecolor("#2b2d31")

        if self.pso is not None:
            plt.suptitle(f"Optimising with {self.results_q.qsize()}/{self.pso.pop} particles for {self.iter}/{self.pso.max_iter+1} iterations")

            iters = len(self.pso.gbest_y_hist)
            if iters > 0:
                cost_graph.set_title(f"Cost over time (current best: {float(self.pso.gbest_y):.3f})")
                cost_graph.set_xlabel("Iteration")
                cost_graph.set_ylabel("Score")
                cost_graph.set_yscale("log")
                cost_graph.get_xaxis().set_major_formatter(plt_ticker.ScalarFormatter())
                cost_graph.get_xaxis().set_minor_formatter(plt_ticker.ScalarFormatter())
                cost_graph.grid(True, "both", "y")
                cost_graph.plot(self.pworst_hist, color='red', label='worst')
                cost_graph.plot(self.pavg_hist, color='grey', label='avg')
                cost_graph.plot(self.pso.gbest_y_hist, color='green', label='best')
                cost_graph.legend()
        
        clients_graph.set_title(f"Client count over time (current: {self.client_data_tracker.get_total_clients()})")
        clients_graph.set_xlabel("Time (minutes)")
        clients_graph.set_ylabel("Clients")
        x = self.client_data_tracker.get_timestamps() + [elapsed/60]
        labels = []
        ys = []
        for ip, counts in self.client_data_tracker.client_counts.items():
            current_count = self.client_data_tracker.client_counts[ip][-1]
            labels.append(self.client_data_tracker.get_client_name(ip) + f" ({current_count})")
            ys.append(counts + [current_count])
        clients_graph.stackplot(x, ys, labels=labels, alpha=.5)
        clients_graph.legend(loc="upper left", fontsize="x-small", reverse=True)

        cpu_graph.set_title("Cpu usage percent over time")
        cpu_graph.set_xlabel("Time (minutes)")
        x = [e[0] for e in self.cpu_percent_hist]
        y = [e[1] for e in self.cpu_percent_hist]
        cpu_graph.grid(True, "both", "y")
        cpu_graph.plot(x, y)
        cpu_graph.set_ybound(lower=0)

        time_per_iter_graph.set_title("Time per iteration")
        time_per_iter_graph.set_xlabel("Iteration")
        time_per_iter_graph.set_ylabel("Time (minutes)")
        time_per_iter_graph.grid(True, "both", "y")
        time_diffs = []
        prev_time = self.start_time
        for timestamp in self.iter_end_times:
            time_diffs.append(timestamp - prev_time)
            prev_time = timestamp
        time_per_iter_graph.plot(time_diffs + [elapsed - prev_time])
        time_per_iter_graph.set_ybound(lower=0)

        # pcplot.plot(coords_graph, self.gbest_x_hist)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)
        plt.clf()


class ClientHandlerThread(threading.Thread):
    _target: MeasureFn

    def __init__(self, init_script: Union[str, None], target: MeasureFn, sock: socket.socket, queue: queue.Queue, results: queue.Queue, init_every: int = 200) -> None:
        super().__init__(None, target, None, [], None, daemon=True)
        self.q = queue
        self.results = results
        self.init_script = init_script
        self.game = TasClient(sock)
        self.init_every = init_every

        self.contributor = "Unknown"

    def run(self):
        total_runs = 0

        # Run tasks, until there is an error
        while True:
            if self.init_every != 0 and total_runs % self.init_every == 0:
                try:
                    self.run_init()
                except Exception as e:
                    logging.exception(f"{self.contributor} client failed to run init script! Dropping.")
                    break

            # Get the next task
            i, point = self.q.get()

            # Try to compute the point
            try:
                score = self._target(self.game, point)
                self.results.put((i, score))
                total_runs += 1
            except Exception as e:
                logging.exception(f"{self.contributor} failed to run a tas")
                self.q.put((i, point))
                self.q.task_done()
                break

            self.q.task_done()
        
        logging.info(f"{self.contributor} exited.")

    def run_init(self):
        if self.init_script is not None:
            self.game.recieve()
            self.game.stop_playback()
            self.game.fast_forward()  # reset ff
            time.sleep(0.1)
            self.game.recieve()
            self.game.start_content_playback(self.init_script)
            self.game.recieve_until(RecvMessageType.PROCESSED_SCRIPT)

        for message in self.game.messages:
            if message.startswith("name: "):
                self.contributor = message[5:30].strip()

        self.game.messages.clear()

    def close(self):
        self.game.sock.close()

class ClientDataTracker:
    def __init__(self) -> None:
        self.timestamps: List[float] = []
        self.client_counts: Dict[str, List[int]] = {} # ip->count
        self.client_names: Dict[str, str] = {} # ip->name

    def set_client_count(self, new_counts: Dict[str, int], timestamp: float):
        self.timestamps.append(timestamp)
        self.timestamps.append(timestamp)
        for client in self.client_counts:
            self.client_counts[client].append(self.client_counts[client][-1])
            self.client_counts[client].append(new_counts.get(client, 0))
        
        for client in new_counts:
            if not self.client_counts.get(client):
                self.client_counts[client] = [0] * (len(self.timestamps)-1)
                self.client_counts[client].append(new_counts[client])
    
    def set_client_name(self, ip: str, name: str):
        if name != "Unknown":
            self.client_names[ip] = name
    
    def get_client_name(self, ip: str) -> str:
        return self.client_names.get(ip, "Unknown")

    def get_total_clients(self) -> int:
        sum = 0
        for counts in self.client_counts.values():
            sum += counts[-1]
        return sum

    def get_client_counts(self) -> Dict[str, int]:
        res = {}
        for ip, count in self.client_counts.items():
            res[self.get_client_name(ip)] = count
        return res

    def get_timestamps(self) -> List[float]:
        return self.timestamps
