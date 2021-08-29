from utils import dotdict

import numpy as np

# how to control your mouse: https://stackoverflow.com/questions/281133/how-to-control-the-mouse-in-mac-using-python
# mouse, keyboard controll: https://pypi.org/project/pynput/
from pynput import mouse as Mouse
from pynput import keyboard as Keyboard

import platform

# Specific system import
if platform.system() == 'Darwin':
    # Mac screen info
    from AppKit import NSScreen
    # Execute applescript
    from subprocess import Popen, PIPE
elif platform.system() == 'Windows':
    import tkinter as tk

class Controller:
    def __init__(self):
        # system
        self.system = platform.system() if platform.system(
        ) != 'Darwin' else 'Mac'  # rename Darwin to Mac
        if self.system == 'Mac':
            self._screen_width, self._screen_hight = self._get_Mac_info()
        elif self.system == 'Windows':
            self._screen_width, self._screen_hight = self._get_Windows_info()
        else:
            print(
                'Warning: The program is not tested on this platform, it may not perform properly.')

        self._mouse = Mouse.Controller()
        self._keyboard = Keyboard.Controller()

        # hyperparameters
        self.MAX_MOUSE_MOVE_SPEED = 100
        self.MAX_SCROLL_SPEED = 5

    def print_system_info(self):
        print(f'Using "{self.system}" platform')
        print(
            f'Screen size (w, h): {(self._screen_width, self._screen_hight)}')

    # Mouse
    def mouse_press(self, button_name):
        if button_name == 'left':
            button = Mouse.Button.left
        elif button_name == 'right':
            button = Mouse.Button.right
        elif button_name == 'middle':
            button = Mouse.Button.middle
        else:
            print(f'Unknown button name: {button_name}')
        self._mouse.press(button)

    def mouse_release(self, button_name):
        if button_name == 'left':
            button = Mouse.Button.left
        elif button_name == 'right':
            button = Mouse.Button.right
        elif button_name == 'middle':
            button = Mouse.Button.middle
        else:
            print(f'Unknown button name: {button_name}')
        self._mouse.release(button)

    def mouse_click(self, button_name):
        if button_name == 'left':
            button = Mouse.Button.left
        elif button_name == 'right':
            button = Mouse.Button.right
        elif button_name == 'middle':
            button = Mouse.Button.middle
        else:
            print(f'Unknown button name: {button_name}')
        self._mouse.click(button, 1)

    def mouse_double_click(self, button_name):
        if button_name == 'left':
            button = Mouse.Button.left
        elif button_name == 'right':
            button = Mouse.Button.right
        elif button_name == 'middle':
            button = Mouse.Button.middle
        else:
            print(f'Unknown button name: {button_name}')
        self._mouse.click(button, 2)

    def mouse_move(self, dx, dy):
        """
        Moves the mouse pointer a number of pixels from its current position.

        Parameters:
        ---
        dx : int
          The horizontal offset.
        dy : int
          The vertical offset.
        """

        x = np.clip(dx, -self.MAX_MOUSE_MOVE_SPEED, self.MAX_MOUSE_MOVE_SPEED)
        y = np.clip(dy, -self.MAX_MOUSE_MOVE_SPEED, self.MAX_MOUSE_MOVE_SPEED)

        # Bound checking
        # WARN:
        # mouse may move to the negative position which used to refer to second monitor,
        # but it also cause mouse move to non-monitor area, so I clip mouse position to
        # keep it in the monitor area.
        # FUTURE: support multi-monitor
        x = np.clip(
            x, 0 - self.mouse_position[0], self.screen_width - self.mouse_position[0])
        y = np.clip(
            y, 0 - self.mouse_position[1], self.screen_hight - self.mouse_position[1])
        self._mouse.move(x, y)

    def scroll(self, dx, dy):
        """
        Sends scroll events.

        Parameters:
        ---
        dx : int
          The horizontal scroll. The units of scrolling is undefined.
        dy : int
          The vertical scroll. The units of scrolling is undefined.
        """
        x = np.clip(dx, -self.MAX_SCROLL_SPEED, self.MAX_SCROLL_SPEED)
        y = np.clip(dy, -self.MAX_SCROLL_SPEED, self.MAX_SCROLL_SPEED)
        self._mouse.scroll(x, y)

    # command
    def switch_desktop_left(self):
        if self.system == 'Mac':
            self._run_applescript('switch_desktop_left')
        elif self.system == 'Windows':
            print('WARNING: if you are using tool (ex. PowerToy) to rebinding ctrl, alt, this function may not work properly')
            self._keyboard.press(Keyboard.Key.cmd)
            self._keyboard.press(Keyboard.Key.ctrl)
            self._keyboard.press(Keyboard.Key.left)
            self._keyboard.release(Keyboard.Key.left)
            self._keyboard.release(Keyboard.Key.ctrl)
            self._keyboard.release(Keyboard.Key.cmd)

    def switch_desktop_right(self):
        if self.system == 'Mac':
            self._run_applescript('switch_desktop_right')
        elif self.system == 'Windows':
            print('WARNING: if you are using tool (ex. PowerToy) to rebinding ctrl, alt, this function may not work properly')
            self._keyboard.press(Keyboard.Key.cmd)
            self._keyboard.press(Keyboard.Key.ctrl)
            self._keyboard.press(Keyboard.Key.right)
            self._keyboard.release(Keyboard.Key.right)
            self._keyboard.release(Keyboard.Key.ctrl)
            self._keyboard.release(Keyboard.Key.cmd)

    def show_control_center(self):
        if self.system == 'Mac':
            self._run_applescript('show_control_center')
        elif self.system == 'Windows':
            with self._keyboard.pressed(Keyboard.Key.cmd):
                self._keyboard.press(Keyboard.Key.tab)
                self._keyboard.release(Keyboard.Key.tab)

    def show_app_expose(self):
        if self.system == 'Mac':
            self._run_applescript('show_app_expose')
        elif self.system == 'Windows':
            pass

    def show_launch_pad(self):
        if self.system == 'Mac':
            # TODO: support this functionality
            pass
        elif self.system == 'Windows':
            self._keyboard.tap(Keyboard.Key.cmd)

    def show_desktop(self):
        if self.system == 'Mac':
            # TODO: support this functionality
            pass
        elif self.system == 'Windows':
            with self._keyboard.pressed(Keyboard.Key.cmd):
                self._keyboard.tap(Keyboard.KeyCode.from_char('d'))

    # getter
    @property
    def mouse_position(self):
        return self._mouse.position

    @property
    def screen_width(self):
        return self._screen_width

    @property
    def screen_hight(self):
        return self._screen_hight

    # listener
    def get_mouse_listener(self, listen_move=True, listen_scroll=True, listen_click=True):
        def on_move(x, y):
            print(f'Pointer move to ({x :.2f}, {y :.2f})')

        def on_scroll(x, y, dx, dy):
            print(
                f'Scrolled ({x :.2f}, {y :.2f}) at {"down" if dy < 0 else "up"}')

        def on_click(x, y, button, pressed):
            print(f'{"Pressed" if pressed else "Released"} at ({x :.2f}, {y :.2f})')
        return Mouse.Listener(
            on_move=(on_move if listen_move else None),
            on_scroll=(on_scroll if listen_scroll else None),
            on_click=(on_click if listen_click else None)
        )

    def get_keyboard_listener(self, listen_press=True, listen_release=True):
        def on_press(key):
            try:
                print(f'alphanumeric key {key.char} pressed')
            except AttributeError:
                print(f'special key {key} pressed')

        def on_release(key):
            print(f'{key} released')
            if key == Keyboard.Key.esc:
                return False

        return Keyboard.Listener(
            on_press=(on_press if listen_press else None),
            on_release=(on_release if listen_release else None)
        )

    # utils
    def _run_applescript(self, script_name):
        APPLESCRIPT_CMD = dotdict({
            'switch_desktop_left': 'tell application "System Events" to key code 123 using control down',
            'switch_desktop_right': 'tell application "System Events" to key code 124 using control down',
            'show_control_center': 'tell application "System Events" to key code 126 using control down',
            'show_app_expose': 'tell application "System Events" to key code 125 using control down',
        })

        if script_name not in APPLESCRIPT_CMD.keys():
            print(f'Unknown script_name: "{script_name}"')
            return

        script = APPLESCRIPT_CMD[script_name]
        p = Popen(['osascript', '-'], stdin=PIPE, stdout=PIPE,
                  stderr=PIPE, universal_newlines=True)
        stdout, stderr = p.communicate(script)
        return p.returncode, stdout, stderr

    def _get_Mac_info(self):
        SCREEN_INDEX = 0
        screen_width = NSScreen.screens()[SCREEN_INDEX].frame().size.width
        screen_hight = NSScreen.screens()[SCREEN_INDEX].frame().size.height
        return screen_width, screen_hight

    def _get_Windows_info(self):
        # IMPROVE: may be better tool to get screen info. It supports multi monitor environments.
        # stackoverflow: https://stackoverflow.com/a/31171430
        # github: https://github.com/rr-/screeninfo

        # screen info
        tk_root = tk.Tk()
        screen_width = tk_root.winfo_screenwidth()
        screen_hight = tk_root.winfo_screenheight()
        return screen_width, screen_hight
