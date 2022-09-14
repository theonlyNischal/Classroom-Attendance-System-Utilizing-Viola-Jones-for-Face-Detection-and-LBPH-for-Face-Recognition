# Import necessary packages

import os
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivy.uix.boxlayout import BoxLayout
from kivymd.toast.kivytoast.kivytoast import toast
from kivy.utils import get_hex_from_color

from kivymd.uix.dialog import  MDDialog
# NavigationDrawer
from kivy.properties import StringProperty

from kivymd.uix.list import OneLineAvatarListItem

class ContentNavigationDrawer(BoxLayout):
    pass


class NavigationItem(OneLineAvatarListItem):
    icon = StringProperty()


class RootWidget(BoxLayout):
    pass

## This is the main app.

class MainApp(MDApp):
    """Main class for the kivy app.
    """
    def __init__(self, **kwargs):
        self.title = "Attendance Management System"
        self.theme_cls = ThemeManager()
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.accent_palette = "Blue"
        self.theme_cls.theme_style="Light"

        super().__init__(**kwargs)
    
    
    def build(self):
        return RootWidget()
    
    def back_to_home_screen(self):
        self.root.ids.student_name.text = ""
        self.root.ids.student_id.text = ""
        self.root.ids.screen_manager.current = "HomeScreen"
    
    def show_ExitDialog(self):
        dialog = MDDialog(
            title="Attendance Management System", 
            text = "Are you [color=%s][b]sure[/b][/color] ?"
            % get_hex_from_color(self.theme_cls.primary_color), 
            size_hint=[.5, .5],
        events_callback=self.stopApp,
        text_button_ok="Exit",
        text_button_cancel="Cancel"
        )
        dialog.open()
    
    def stopApp(self, text_of_selection, popup_widget):
        
        if text_of_selection == "Exit":
            self.stop()
        else:
            pass
    
    def performAttendance(self):
        os.system("python faceRecognition.py")
    
    def captureTrainingImages(self, student_name, student_id, screen_manager):
        if len(student_name) > 0 and len(student_name) <= 23 and len(student_id) > 0 and len(student_id) <= 6:
            os.system("python captureTrainingImages.py {} {}".format(student_name, student_id))
            toast("Training Images Collected.")
            screen_manager.current = "HomeScreen"
        else:
            toast("Please Enter Correct Details.")
     
    def recordAttendance(self):
        """Calls a recordAttendance.py to record the attendacne.
        """
        os.system("python recordAttendance.py")
    

    


MainApp().run()
