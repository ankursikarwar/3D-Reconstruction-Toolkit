
# from tkinter import *
from Tkinter import *
import tkFileDialog
from tkFileDialog import askdirectory
from PIL import Image, ImageTk
import sys, os
import math
import pickle
import numpy as np
from smpl.hello_smpl import get_mesh
root = Tk()
from smpl.smpl_webuser.serialization import load_model

from hmr.democheck2 import rerenders

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "3"


m = load_model('smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
root.title('Mesh GUI')
root.geometry('1024x1024')
bottomframe = Frame(root)
leftframe = Frame(root)
leftframe.pack(side= LEFT)
rightframe = Frame(root)
rightframe.pack(side= RIGHT)
bottomframe.pack(side=BOTTOM)

Posex = dict()       #dict for shape and pose parameters
Posey = dict()
Posez = dict()
Beta = dict()

im_path =""
params_path = ""


for i in range(1,25):   #initializing shape and pose parameters   
    Posex[str(i)] = 0
    Posey[str(i)] = 0
    Posez[str(i)] = 0

for i in range(1, 11):
    Beta[str(i)] = 0

class scale:
    def __init__(self,master,text,n):
        frame = Frame(master)
        frame.pack(side=TOP)
        self.v = DoubleVar()    #shape and pose param values
        self.v1 = IntVar()
        self.v2 = StringVar()
        self.v3 = StringVar()
        self.v4 = StringVar()
        self.s = Scale(frame,from_=-3.14159,to=3.14159,orient=HORIZONTAL,resolution=0.0001,length=170,width=4,variable=self.v)
        self.s.grid(row=n+1,column=1)   #scaling grid for pose params
        self.l = Label(frame,text=text)     
        self.l.grid(row=n+1,column=0)   #Serial numbering for pose paramas
        self.rx = Radiobutton(frame,text='x',value = 1,variable=self.v1)
        self.rx.grid(row=n+1,column=2)  #x RadioButton
        self.ry = Radiobutton(frame,text='y',value = 2,variable=self.v1)
        self.ry.grid(row=n+1,column=3)  #y RadioButton
        self.rz = Radiobutton(frame,text='z',value = 3,variable=self.v1)
        self.rz.grid(row=n+1,column=4)  #z RadioButton


        self.no = Radiobutton(frame,text='NO',value = 4,variable=self.v1)
        self.no.grid(row=n+1,column=5)  #z RadioButton        
        

        self.lx = Label(frame,textvariable=self.v2, font='TkFixedFont')
        self.lx.grid(row=n+1,column=6)
        self.ly = Label(frame,textvariable=self.v3)
        self.ly.grid(row=n+1,column=7)
        self.lz = Label(frame,textvariable=self.v4)
        self.lz.grid(row=n+1,column=8)


class scaleS:
    def __init__(self,master,text,n):
        frame = Frame(master)
        frame.pack(side=TOP)
        self.v = DoubleVar()
        self.s = Entry(frame)
        self.s.grid(row=n,column=2)
        self.shape = Label(frame,textvariable=self.v)
        self.shape.grid(row=n,column=3)
        self.l = Label(frame,text=text)
        self.l.grid(row=n,column=1)

class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        w,h = 650, 650
        self.pack()
        self.file = Button(self, text='Browse', command=self.choose)
        self.choose = Label(self, text="Choose file").pack()
        self.image = ImageTk.PhotoImage(Image.open("b.jpg"))
        self.label = Label(image=self.image)
        

        self.file.pack()
        self.label.pack()


    def choose(self):
        global im_path, params_path
        ifile = tkFileDialog.askopenfile(parent=self,mode='rb',title='Choose a file')
        im_path = ifile.name
        cur_path = os.getcwd()
        hmr_path = os.path.join((cur_path,"/hmr"))    
        os.chdir(hmr_path)
        cmd = "python -m demo1 --img_path " + im_path
        print cmd
        os.system(cmd)
        os.chdir(cur_path)
        self.load_image()
    
    def load_image(self, flag=0):
        global im_path, params_path
        cur_path = os.getcwd()
        img_path = "/test_img/" + im_path.split('/')[-1][:-4] + ".png"
        img_path = os.path.join((cur_path,img_path))
        params_path = "/test_params/" + im_path.split('/')[-1][:-4] + ".pkl"
        params_path = os.path.join((cur_path,params_path))
        print(im_path, img_path, params_path)
        self.image2 = ImageTk.PhotoImage(Image.open(img_path).resize((500, 500)))
        self.label.configure(image=self.image2)
        self.label.image=self.image2
        self.load_params(params_path, flag)

    def load_params(self, path, flag):
        if flag is not 0:
            print(1,poses_dict[0].v2.get())
            self.show_params()
            return
        file = open(path, 'rb')
        data = pickle.load(file)
        pose = data['pose'].reshape(72,)
        shape = data['shape'].reshape(10,)
        print pose.shape
        for i in range(0, 10):
            Beta[str(i+1)] = shape[i]
        for i in range(0, 72):
            if i%3==0:
                Posex[str(i/3+1)] = pose[i]
            elif i%3==1:
                Posey[str(i/3+1)] = pose[i]
            elif i%3==2:
                Posez[str(i/3+1)] = pose[i]
        self.show_params()


    def show_params(self):
        for i in range(0,24):
            poses_dict[i].v2.set(str(Posex[str(i+1)])[:6])
            poses_dict[i].v3.set(str(Posey[str(i+1)])[:6])
            poses_dict[i].v4.set(str(Posez[str(i+1)])[:6])
        for i in range(0, 10):
            beta_dict[i].v.set(str(Beta[str(i+1)]))
            beta_dict[i].s.delete(0, END)
            beta_dict[i].s.insert(0, str(Beta[str(i+1)]))


l2 = Label(root,text='Shape Params')
l2.pack()
b1 = scaleS(root,'1',1)
b2 = scaleS(root,'2',2)
b3 = scaleS(root,'3',3)
b4 = scaleS(root,'4',4)
b5 = scaleS(root,'5',5)
b6 = scaleS(root,'6',6)
b7 = scaleS(root,'7',7)
b8 = scaleS(root,'8',8)
b9 = scaleS(root,'9',9)
b10 = scaleS(root,'10',10)
l = Label(leftframe,text='Pose Params')
l.pack()
beta_dict = []

beta_dict.append(b1)
beta_dict.append(b2)
beta_dict.append(b3)
beta_dict.append(b4)
beta_dict.append(b5)
beta_dict.append(b6)
beta_dict.append(b7)
beta_dict.append(b8)
beta_dict.append(b9)
beta_dict.append(b10)

s1 = scale(leftframe,'1',1)
s2 = scale(leftframe,'2',2)
s3 = scale(leftframe,'3',3)
s4 = scale(leftframe,'4',4)
s5 = scale(leftframe,'5',5)
s6 = scale(leftframe,'6',6)
s7 = scale(leftframe,'7',7)
s8 = scale(leftframe,'8',8)
s9 = scale(leftframe,'9',9)
s10 = scale(leftframe,'10',10)
s11 = scale(leftframe,'11',11)
s12 = scale(leftframe,'12',12)
s13 = scale(leftframe,'13',13)
s14 = scale(leftframe,'14',14)
s15 = scale(leftframe,'15',15)
s16 = scale(leftframe,'16',16)
s17 = scale(leftframe,'17',17)
s18 = scale(leftframe,'18',18)
s19 = scale(leftframe,'19',19)
s20 = scale(leftframe,'20',20)
s21 = scale(leftframe,'21',21)
s22 = scale(leftframe,'22',22)
s23 = scale(leftframe,'23',23)
s24 = scale(leftframe,'24',24)
poses_dict = []

poses_dict.append(s1)
poses_dict.append(s2)
poses_dict.append(s3)
poses_dict.append(s4)
poses_dict.append(s5)
poses_dict.append(s6)
poses_dict.append(s7)
poses_dict.append(s8)
poses_dict.append(s9)
poses_dict.append(s10)
poses_dict.append(s11)
poses_dict.append(s12)
poses_dict.append(s13)
poses_dict.append(s14)
poses_dict.append(s15)
poses_dict.append(s16)
poses_dict.append(s17)
poses_dict.append(s18)
poses_dict.append(s19)
poses_dict.append(s20)
poses_dict.append(s21)
poses_dict.append(s22)
poses_dict.append(s23)
poses_dict.append(s24)

# print poses_dict

app = GUI(rightframe)

def getVal():
    global frame_count
    global total_frames
    global video_path
    global video_path_hmr
    global video_hmr_pickle
    for i in range(0, 24):

        if poses_dict[i].v1.get() == 1:
            Posex[str(i+1)] = poses_dict[i].v.get()
        elif poses_dict[i].v1.get() == 2:
            Posey[str(i+1)] = poses_dict[i].v.get()
        elif poses_dict[i].v1.get() == 3:
            Posez[str(i+1)] = poses_dict[i].v.get()
    
    for i in range(0, 10):
        Beta[str(i+1)] = beta_dict[i].v.get()
        Beta[str(i+1)] = beta_dict[i].s.get()
        # print(beta_dict[i].s.get())

    pose = []
    beta = []
    for i in range(0, 24):
        pose.append(Posex[str(i+1)])
        pose.append(Posey[str(i+1)])
        pose.append(Posez[str(i+1)])
    for i in range(0, 10):
        beta.append(Beta[str(i+1)])
    # print(pose, beta)

    #print(params_path)
    hmr_params_path = video_hmr_pickle + str(frame_count) + ".pkl"
    print(hmr_params_path)
    verts = get_mesh(m, np.array(pose), np.array(beta))
    file = open(hmr_params_path, 'rb')
    data = pickle.load(file)
    print(np.array(beta))
    print(type(data['shape'][0][2]))
    proc_param = data['proc_param']
    cam = data['cam']
    joints = data['joints']
    frame_path = video_path + str(frame_count) + ".png"
    print(frame_path)
    folder_name = frame_path.split('/')[-2]
    rerenders(frame_path, proc_param, joints, verts, cam, folder_name)
    cur_path = os.getcwd()
    hmr_frame_path = cur_path + "/frames/FRAMES_HMR/" + folder_name + '/' + frame_path.split('/')[-1][:-4] + ".png"
    

    params = {'proc_param': proc_param, 'joints': joints, 'cam': cam, 'pose': np.array(pose), 'shape': np.array(beta)}
    print(params)
    print(hmr_params_path)
    with open(hmr_params_path, 'wb') as f:
        pickle.dump(params, f)


    app.load_params(hmr_params_path, flag = 1)
    app.image2 = ImageTk.PhotoImage(Image.open(hmr_frame_path).resize((500, 500)))
    app.label.configure(image=app.image2)

    
def imp_vid():
    global video_path
    global video_path_hmr
    global video_hmr_pickle
    ifile = tkFileDialog.askopenfile(mode='rb',title='Choose a file')
    vid_path = ifile.name
    print(vid_path)
    if not os.path.exists('frames/FRAMES_WITHOUT_HMR/'+vid_path.split('/')[-1][:-4]):
         os.makedirs('frames/FRAMES_WITHOUT_HMR/'+vid_path.split('/')[-1][:-4])
    if not os.path.exists('frames/FRAMES_HMR/'+vid_path.split('/')[-1][:-4]):
         os.makedirs('frames/FRAMES_HMR/'+vid_path.split('/')[-1][:-4])
    if not os.path.exists('frames/FRAMES_PICKLE/'+vid_path.split('/')[-1][:-4]):
         os.makedirs('frames/FRAMES_PICKLE/'+vid_path.split('/')[-1][:-4])


    cur_path = os.getcwd()
    video_path = cur_path + '/frames/FRAMES_WITHOUT_HMR/' +vid_path.split('/')[-1][:-4] + '/'
    video_path_hmr = cur_path + '/frames/FRAMES_HMR/' +vid_path.split('/')[-1][:-4] + '/'
    video_hmr_pickle = cur_path + '/frames/FRAMES_PICKLE/' +vid_path.split('/')[-1][:-4] + '/'
    print(video_path)
    print(video_path_hmr)
    print(video_hmr_pickle)
    cmd = "ffmpeg -i " + vid_path + " -r 1 " + video_path +"$filename%d.png"
    print cmd
    os.system(cmd)
    load_first()

def load_first():
    # global im_path

    global frame_count
    global total_frames
    global video_path
    global video_path_hmr
    global video_hmr_pickle

    frame_count = 1
    _, _, files = next(os.walk(video_path))
    total_frames = len(files)
    frame_path = video_path + str(frame_count) + ".png"
    hmr_params_path = video_hmr_pickle + str(frame_count) + ".pkl"
    cur_path = os.getcwd()    
    hmr_path = cur_path + "/hmr"
    os.chdir(hmr_path)
    cmd = "python -m democheck2 --img_path " + frame_path
    print cmd



    if not(os.path.isfile(hmr_params_path)):
        os.system(cmd)
    print "hello"
    os.chdir(cur_path)
    hmr_frame_path = video_path_hmr + str(frame_count) + ".png"
    print(frame_path, hmr_frame_path, hmr_params_path)    
    app.load_params(hmr_params_path, flag = 0)
    app.image2 = ImageTk.PhotoImage(Image.open(hmr_frame_path).resize((500, 500)))
    app.label.configure(image=app.image2)

    tot_frame.set("Total Frames : " + str(total_frames))
    frame_input.delete('1.0', END)
    frame_input.insert(END, str(frame_count))
    # frame_count_variable.set("Present frame : " + str(frame_count))

    while frame_count < total_frames:
        frame_count+=1
        # frame_count_variable.set("Present frame : " + str(frame_count))
        # label_framecount.pack(side=BOTTOM).
        # label_framecount["text"] = "Present frame : " + str(frame_count)
        print(frame_count)
        # print(total_frames)
        frame_path = video_path + str(frame_count) + ".png"
        hmr_path = "/home/ankur/GUI_project/hmr"
        cur_path = os.getcwd()
        os.chdir(hmr_path)
        cmd = "python -m democheck2 --img_path " + frame_path
        print cmd
        os.system(cmd)
        print "hello"
        os.chdir(cur_path)



    # app.label.image=app.image2
    # app.load_params(params_path, flag)

def next_frame():
    global frame_count
    global total_frames
    global video_path
    global video_path_hmr
    global video_hmr_pickle
    if frame_count < total_frames:
        frame_count+=1
    print(frame_count)
    # print(total_frames)
    frame_path = video_path + str(frame_count) + ".png"
    cur_path = os.getcwd()
    hmr_path = cur_path + "/hmr"
    os.chdir(hmr_path)
    cmd = "python -m democheck2 --img_path " + frame_path
    print cmd
    hmr_params_path = video_hmr_pickle + str(frame_count) + ".pkl"        
    if not(os.path.isfile(hmr_params_path)):
        os.system(cmd)
    print "hello"
    os.chdir(cur_path)
    hmr_frame_path = video_path_hmr + str(frame_count) + ".png"
    print(frame_path, hmr_frame_path, hmr_params_path)    
    app.load_params(hmr_params_path, flag = 0)
    app.image2 = ImageTk.PhotoImage(Image.open(hmr_frame_path).resize((500, 500)))
    app.label.configure(image=app.image2)
    frame_input.delete('1.0', END)
    frame_input.insert(END, str(frame_count))

def previous_frame():
    global frame_count
    global total_frames
    global video_path
    global video_path_hmr
    global video_hmr_pickle
    if frame_count>1:
        frame_count-=1
        print(frame_count)
        # print(total_frames)
        frame_path = video_path + str(frame_count) + ".png"
        cur_path = os.getcwd()
        hmr_path = cur_path + "/hmr"
        os.chdir(hmr_path)
        cmd = "python -m democheck2 --img_path " + frame_path
        print cmd
        hmr_params_path = video_hmr_pickle + str(frame_count) + ".pkl"    
        if not(os.path.isfile(hmr_params_path)):
            os.system(cmd)
        print "hello"
        os.chdir(cur_path)
        hmr_frame_path = video_path_hmr + str(frame_count) + ".png"
        hmr_params_path = video_hmr_pickle + str(frame_count) + ".pkl"
        print(frame_path, hmr_frame_path, hmr_params_path)    
        app.load_params(hmr_params_path, flag = 0)
        app.image2 = ImageTk.PhotoImage(Image.open(hmr_frame_path).resize((500, 500)))
        app.label.configure(image=app.image2)
        frame_input.delete('1.0', END)
        frame_input.insert(END, str(frame_count))

def goto_frame():
    global frame_count
    global total_frames
    global video_path
    global video_path_hmr
    global video_hmr_pickle
    input = frame_input.get("1.0",END)
    frame_count = int(input)
    print(frame_count)
    if (frame_count<=total_frames) and (frame_count>0):
        frame_path = video_path + str(frame_count) + ".png"
        cur_path = os.getcwd()
        hmr_path = cur_path + "/hmr"
        os.chdir(hmr_path)
        cmd = "python -m democheck2 --img_path " + frame_path
        print cmd
        hmr_params_path = video_hmr_pickle + str(frame_count) + ".pkl"
        if not(os.path.isfile(hmr_params_path)):
            os.system(cmd)
        print "hello"
        os.chdir(cur_path)
        hmr_frame_path = video_path_hmr + str(frame_count) + ".png"
        print(frame_path, hmr_frame_path, hmr_params_path)    
        app.load_params(hmr_params_path, flag = 0)
        app.image2 = ImageTk.PhotoImage(Image.open(hmr_frame_path).resize((500, 500)))
        app.label.configure(image=app.image2)


# def check(n, master):
#     global frame_count
#     global frame_count_check
#     frame_count = 1
#     frame_count_check = 1
#     if (frame_count != frame_count_check):
#         frame_count_check = frame_count
#         best.set("Present frame : " + str(frame_count))









button2 = Button(bottomframe,text = 'Refresh',fg = 'red',command=getVal)
button2.pack()
# save = Button(bottomframe, text = 'Save', fg = 'red')
# save.pack()
videoimp = Button(bottomframe, text= 'Import Video', fg = 'red',command=imp_vid)
videoimp.pack(side = TOP)

tot_frame = StringVar()
tot_frame.set("Total Frames : 0")
label_frame = Label(bottomframe, textvariable=tot_frame)
label_frame.pack()

go_button = Button(bottomframe, text='Go', fg = 'red',command=goto_frame)
go_button.pack(side = RIGHT)

next_button = Button(bottomframe, text= 'Next Frame', fg = 'blue',command=next_frame)
next_button.pack(side = RIGHT)
previous_buton = Button(bottomframe, text= 'Previuos Frame', fg = 'blue',command=previous_frame)
previous_buton.pack(side = LEFT)
img2 = ImageTk.PhotoImage(Image.open('b.jpg').resize((244, 244)))
label2 = Label(rightframe,image=img2)
label2.pack()
frame_input = Text(bottomframe, height = 1, width = 5)
frame_input.pack(side = LEFT)



root.mainloop()





