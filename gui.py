import os.path
import tkinter as tk
import tkinter.messagebox
import pynput.keyboard as keyboard
from capture_data import RealTimeKeyListener, OfflineListener
import tkinter.font as font
from tkinter import filedialog

title = "Listener by BM"

pl_message = ''' 
Na podstawie art. 6 ust. 1 lit. a, art. 9 ust. 2 lit. a rozporządzenia Parlamentu Europejskiego i Rady (UE) 2016/679 
z dnia 27 kwietnia 2016 r. w sprawie ochrony osób fizycznych w związku z przetwarzaniem danych osobowych i w sprawie 
swobodnego przepływu takich danych oraz uchylenia dyrektywy 95/46/WE (ogólne rozporządzenie o ochronie danych) 
Dz. Urz. UE L 119/1, z 4.5.2016, zwanego dalej „RODO” wyrażam zgodę na przetwarzanie następujących kategorii moich 
danych osobowych (identyfikatory internetowe), w zakresie badań zaprojektowanego narzędzia w celach naukowych. 
Podanie przeze mnie danych osobowych jest dobrowolne. Podane przeze mnie dane osobowe będą przetwarzane wyłącznie w celach naukowych. 
Jest mi wiadomym, że posiadam  prawo do:
1)	żądania od wskazanego w niniejszym oświadczeniu administratora danych osobowych:
a)	dostępu do moich danych osobowych;
b)	sprostowania moich danych osobowych;
c)	usunięcia moich danych osobowych, jeżeli zachodzi jedna z okoliczności wskazanych w art. 17 ust. 1 RODO i jeżeli przetwarzanie moich danych osobowych nie jest niezbędne w zakresie wskazanym w art. 17 ust. 3 RODO;
d)	ograniczenia przetwarzania moich danych osobowych w przypadkach wskazanych w art. 18 ust. 1 RODO,  
2)	wniesienia do wskazanego w niniejszym oświadczeniu administratora danych osobowych sprzeciwu wobec przetwarzania moich danych osobowych:
a)	na potrzeby marketingu bezpośredniego, w tym profilowania, w zakresie, w jakim przetwarzanie jest związane z takim marketingiem bezpośrednim,
b)	do celów badań naukowych lub historycznych lub do celów statystycznych na mocy art. 89 ust. 1 RODO, z przyczyn związanych z moją szczególną sytuacją, chyba że przetwarzanie jest niezbędne do wykonania zadania realizowanego 
w interesie publicznym.
3)	przenoszenia moich danych osobowych,
4)	wniesienia skargi do organu nadzorczego, tj. do Prezesa Urzędu Ochrony Danych Osobowych, w przypadku uznania, że przetwarzanie moich danych osobowych narusza przepisy RODO,
5)	wycofania w dowolnym momencie zgody na przetwarzanie moich danych osobowych. 
'''

en_msg = """
The controller of your personal data is Bartłomiej Marek (individual or legal entity). 
Pursuant to Art. 37 (1) (a) GDPR, the controller of your personal data has appointed the Data Protection Officer whom you may contact in matters related to the processing of personal data. 
Your personal data will be processed only for scientific purposes pursuant to Art. 6 (1) (a) GDPR.
The recipients of your personal data may be entities authorized under the law.
Your personal data will be kept until the consent is withdrawn or the purpose ceases to exist.
You have the right to access your data and, subject to the provisions of the law, you have the right to:
withdraw the consent granted, with the proviso that the withdrawal of consent will not affect the lawfulness of the processing which was carried out on the basis of your consent before its withdrawal;
- rectify your data;
- delete your data;
- restrict your data processing;
- object to the processing of personal data.
When you feel that the processing of personal data violates generally applicable provisions in this regard you have the right to lodge a complaint with the competent supervisory body, i.e. the President of the Office for Personal Data Protection, if you think that the processing of your personal data violates generally applicable regulations in this respect.
Providing personal data is voluntary and you are not obliged to provide it. The consequence of not providing them will be the inability to receive an offer of studies at the Białystok University of Technology.
Personal data will not be used for automated decision making or profiling referred to in art. 22.
"""


def on_online_start():
    global online_listener, file_name, title
    if online_listener:
        tkinter.messagebox.showwarning(title=title, message="Listener IS running!")
    else:
        online_listener = RealTimeKeyListener()
        if file_name is not None:
            online_listener.destination_json_file_path = file_name[:-5] + "_online" + file_name[-5:]
        tkinter.messagebox.showwarning(title=title, message="Listener IS running! Your destination file is: " +
                                                            str(online_listener.destination_json_file_path))
def confirm_nickname():
    global file_name, nickname_entry
    nickname = nickname_entry.get()
    file_name = get_destination_file_name(nickname)
    tkinter.messagebox.showinfo(title="Nickname Confirmed", message=f"Destination file changed to: {file_name}")


def press_end_combination():
    controller = keyboard.Controller()
    controller.press(keyboard.Key.esc)
    controller.press(keyboard.Key.f4)


def online_stop():
    global online_listener
    if isinstance(online_listener, RealTimeKeyListener):
        tkinter.messagebox.showwarning(title=title,
                                       message="Listener has just been stopped. \nYou can start it again, "
                                               "quit or change destination file path")
        press_end_combination()
        press_end_combination()
        online_listener.keyboard_listener.join()
        online_listener.mouse_listener.join()
        online_listener = None
    else:
        tkinter.messagebox.showwarning(title=title, message="Listener IS NOT running!")


def offline_start():
    global file_name
    offline_lstnr = OfflineListener()
    file_selected = filedialog.askopenfilename(filetypes=[('TXT', '*.txt'), ('PDF', '*pdf'), ('DOCX', '*docx')])
    offline_lstnr.source_txt_file_path = file_selected
    if file_name is not None:
        offline_lstnr.destination_json_file_path = file_name
    tkinter.messagebox.showinfo(title=title,
                                message="You have just chosen" + str(file_selected)
                                        + "\nDestination_file is: " + offline_lstnr.destination_json_file_path)
    text = offline_lstnr.read_text_file()
    if text:
        offline_lstnr.write_to_json_file(text)


def agreement(message_title, message_text):
    msg_box = tkinter.messagebox.askyesno(title=message_title, message=message_text)
    if not msg_box:
        root.destroy()


def non_agreement(message_title, message_text):
    msg_box = tkinter.messagebox.askyesno(title=message_title, message=message_text)
    if msg_box:
        root.destroy()


def non_ex_agreement():
    global online_listener
    ex_title = 'Exit application'
    ex_msg = 'Are you sure that you want to exit the application?'
    if online_listener:
        online_stop()
    non_agreement(ex_title, ex_msg)


def get_destination_file_name(nickname):
    if nickname:
        return nickname + '.json'
    else:
        return 'destination.json'


# Modify the set_path function
def set_path():
    global online_listener, file_name, offline_lstnr, nickname_entry
    folder_selected = filedialog.askdirectory()
    nickname = nickname_entry.get()  # Get the nickname from the Entry widget
    file_name = os.path.join(folder_selected, get_destination_file_name(nickname))

    if isinstance(online_listener, RealTimeKeyListener) and file_name:
        online_listener.destination_json_file_path = file_name[:-5] + "_online" + file_name[-5:]
        print(online_listener.destination_json_file_path)
    if isinstance(offline_lstnr, OfflineListener) and file_name:
        offline_lstnr.destination_json_file_path = file_name[:-5] + "_offline" + file_name[-5:]


if __name__ == '__main__':
    try:
        online_listener, file_name, offline_lstnr = None, None, None
        background = '#2C3E50'
        button_colour = "#3498DB"
        active_button_colour = "#2980B9"
        text_color = "#ECF0F1"

        root = tk.Tk()
        root.geometry('800x900')
        root.resizable(True, True)
        root.title(title)
        root.configure(background=background)

        myFont = font.Font(weight='bold', family='Century Gothic', size=12)
        titleFont = font.Font(weight='bold', family='Century Gothic', size=16)

        # Title Label
        titleLabel = tk.Label(root, text="Listener by BM", bg=background, fg=text_color, font=titleFont)
        titleLabel.pack(pady=20)

        nickname_label = tk.Label(root, text="Enter Nickname:", bg=background, fg=text_color, font=myFont)
        nickname_label.pack(pady=10)
        nickname_entry = tk.Entry(root, bg="white", fg="black", width=30, font=myFont)
        nickname_entry.pack(pady=10)

        confirm_nickname_button = tk.Button(root, command=confirm_nickname, text="Confirm Nickname", bg=button_colour,
                                            fg=text_color, width=25, height=2, activebackground=active_button_colour,
                                            font=myFont)
        confirm_nickname_button.pack(pady=15)

        # Start Button
        start_button = tk.Button(root, command=on_online_start, text="Start Online Listener", bg=button_colour,
                                 fg=text_color, width=25, height=2, activebackground=active_button_colour, font=myFont)
        start_button.pack(pady=15)

        # Stop Button
        stop_button = tk.Button(root, command=online_stop, text="Stop Online Listener", bg=button_colour, fg=text_color,
                                width=25, height=2, activebackground=active_button_colour, font=myFont)
        stop_button.pack(pady=15)

        # Offline Button
        offline_btn = tk.Button(root, command=offline_start, text="Offline Listener", bg=button_colour, fg=text_color,
                                width=25, height=2, activebackground=active_button_colour, font=myFont)
        offline_btn.pack(pady=15)

        # Set Path Button
        set_path_button = tk.Button(root, command=set_path, text="Set path for end files", bg=button_colour, fg=text_color,
                                    width=25, height=2, activebackground=active_button_colour, font=myFont)
        set_path_button.pack(pady=15)

        # Exit Button
        exit_button = tk.Button(root, text='Exit', command=non_ex_agreement, bg=button_colour, fg=text_color, width=25,
                                height=2, activebackground=active_button_colour, font=myFont)
        exit_button.pack(pady=15)

        agreement('GDPR clause', en_msg)

        root.mainloop()
    except:
        pass


