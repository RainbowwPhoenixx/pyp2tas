import server

game = server.TasServer("127.0.0.1", 6555)

last_tas = ""

HELP_STR= """
Actions:
* connect
    Attempt to connect to the game
* play <filename>
    Play the given tas file
* replay
    Replay the last tas file
* stop
* rate <speed>
* resume
* pause
* ff <tick> <pause_after>
    Fast-forward to the given tick
* pauseat <tick>
* entityinfo <entity selector string>
    Request entity info for the given entity
* recv
    Read data from the server and display the results
* exit
    Exit this program
* help
    Display this

Some arguments may be optionnal.
"""

while True:

    command = ""
    while command.strip() == "":
        command = input("Enter an action> ")

    argv = command.strip().split(" ")
    argc = len(argv)

    # print(argv)

    try:
        if argv[0] == "connect":
            game.connect()
        elif argv[0] == "play":
            game.start_file_playback(argv[1])
            last_tas = argv[1]
        elif argv[0] == "replay":
            game.start_file_playback(last_tas)
        elif argv[0] == "stop":
            game.stop_playback()
        elif argv[0] == "rate":
            game.change_playback_speed(float(argv[1]))
        elif argv[0] == "resume":
            game.resume_playback()
        elif argv[0] == "pause":
            game.pause_playback()
        elif argv[0] == "ff":
            if argc == 1:
                game.fast_forward()
            elif argc == 2:
                game.fast_forward(int(argv[1]))
            elif argc == 3:
                game.fast_forward(int(argv[1]), bool(argv[2]))
        elif argv[0] == "pauseat":
            game.pause_at(int(argv[1]))
        elif argv[0] == "advance" or argv[0] == "a":
            game.advance_playback()
        elif argv[0] == "entityinfo":
            if argc == 1:
                game.entity_info()
            elif argc == 2:
                game.entity_info(argv[1])
        elif argv[0] == "recv":
            entity_info = game.recieve()
            if len(entity_info):
                print(str(entity_info[0]))
            print(str(game))
        elif argv[0] == "exit":
            exit(0)
        elif argv[0] == "help":
            print(HELP_STR)
        else:
            print("Unknown command")
    
    except IndexError:
        print("Not enough arguments for command " + argv[0])
