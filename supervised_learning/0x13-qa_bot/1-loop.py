#!/usr/bin/env python3
""" Create the loop """
import cmd


class QA_bot(cmd.Cmd):
    """ class QA_bot """

    prompt = 'Q: '

    def onecmd(self, arg):
        """ onecmd override """
        return cmd.Cmd.onecmd(self, arg.lower())

    def do_EOF(self, arg):
        """ default method. """
        print("A:")

    def default(self, arg):
        """ default method. """
        print("A:")

    def do_exit(self, arg):
        """ exit method. """
        print("A:", "Goodbye")
        return True

    def do_quit(self, arg):
        """ quit method. """
        print("A:", "Goodbye")
        return True

    def do_goodbye(self, arg):
        """ goodbye method. """
        print("A:", "Goodbye")
        return True

    def do_bye(self, arg):
        """ bye method. """
        print("A:", "Goodbye")
        return True


if __name__ == '__main__':
    QA_bot().cmdloop()
