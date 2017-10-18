from Tkinter import *
from tkFileDialog import askopenfilename, asksaveasfilename
from PIL import ImageTk, Image
import numpy as np
from math import sqrt, ceil
from detectcrack import generate_ridges
from colorseg import generate_segments
from thresholding import threshold_segments, apply_threshold_segments
from scipy import ndimage

__version__ = "v1.09"

window = Tk()
window.title("Crack Detection " + __version__)
window.iconbitmap("favicon.ico")
class SliderObject(object):

    def __init__(self, context, text=None, minVal=0, maxVal=1, step=1):
        
        self.base = Frame(context, height=2, bd=2)
        
        self.sliderLabel = Label(self.base, text=text, width=11, anchor=E)
        self.sliderLabel.pack(side="left")

        if isinstance(step, int):
            self.var = IntVar()
        elif isinstance(step, float):
            self.var = DoubleVar()

        self.slider = Scale(self.base, from_=minVal, to=maxVal, length=120, 
         orient=HORIZONTAL, showvalue=0, command=self.var.set, resolution=step)
        self.slider.pack(side="left")

        self.entervalue = Entry(self.base, textvariable=self.var, width=10)
        self.entervalue.pack(side="left")

        self.base.pack()

        def setSlider(*args):
            try:
                self.slider.set(self.var.get())
            except ValueError:
                pass

        self.var.trace('w',setSlider)
        
    def getParameter(self):
        return self.var.get()

    def enable(self):
        self.slider.configure(state="normal")
        self.entervalue.configure(state="normal")

    def disable(self):
        self.slider.configure(state="disabled")
        self.entervalue.configure(state="disabled")

    def hide(self):
        self.base.pack_forget()

    def show(self):
        self.base.pack()


class parameters(object):

    def __init__(self):

        self.kernel_size = None
        self.angles = None
        self.S = None
        self.A = None
        self.k = 1
        self.resolution = 1
        self.T = np.zeros((2, 1))
        self.T_new = np.zeros((2, 1))
        self.T_scale = np.zeros((1,))

class imageData(object):

    def __init__(self, im):
        
        self.im = np.array(im)
        self.image = Image.fromarray(im)
        self.ridge_disp = np.zeros(self.im.shape[:2])
        self.ridge = np.zeros(self.im.shape[:2])
        self.segments_disp = np.zeros(self.im.shape[:2])
        self.segments = np.zeros(self.im.shape[:2])
        self.crackmap = np.zeros(self.im.shape[:2])
        self.crackmap_disp = np.zeros(self.im.shape[:2])
        self.crackmap_dial = np.zeros(self.im.shape[:2])
        self.crackmap_dilated = np.zeros(self.im.shape[:2])
        self.overlay = np.zeros(self.im.shape)
        self.overlay_dilated = np.zeros(self.im.shape)
        self.display = None
        self.disp_ratio = 1
        self.parameters = parameters()
        self.state = "Blank"
        self.mask = np.ones(self.im.shape[:2])
        self.r = int(0.5 * sqrt(sqrt(self.image.width*self.image.height)))
        print self.r
        self.brush = 1 

def setFileName(*args):
    fileName.set(askopenfilename())

def segmentImage():
    try:
        global data
        p = data.parameters
        image = data.image
        im = data.im
    except NameError:
        return

    k = slider_clust.getParameter()
    resolution = slider_resol.getParameter()
    segments = generate_segments(image, k, resolution).astype(int)

    segments_disp = np.zeros(segments.shape+(3,))
    for seg in range(k):
        for i in range(3):
            c = np.sum(im[:,:,i][seg==segments])/np.sum(seg==segments)
            segments_disp[:,:,i][seg==segments] = c
    segments_disp = segments_disp.astype('uint8')
    
    dropvar.set('Segment 1')
    drop['menu'].delete(0, 'end')
    options = ['Segment'+str(i+1) for i in range(k)]
    for option in options:
        drop['menu'].add_command(label=option, command=lambda choice=option: dropvar.set(choice))

    data.segments = segments
    data.segments_disp = segments_disp
    p.k = k
    p.resolution = resolution
    p.T = np.zeros((2, k))
    p.T_new = np.zeros((2, k))
    p.T_scale = np.zeros((k,))

    if data.ridge.any():
        autoParameter(data)
    setPanelState(segments_disp,"SegOverview")

def detectCracks():
    try:
        global data
        p = data.parameters
        im = data.im
    except NameError:
        return
    
    angles = slider_angle.getParameter()
    size = slider_csize.getParameter()
    if advanced2.get():
        S = slider_S.getParameter()
        A = slider_A.getParameter()
        if S > A:
            S, A = A, S
    else:
        S = size
        A = size     
    
    ridge = generate_ridges(im, angles, S, A, crackType.get())
    ridge_disp = ridge.clip(min=0)
    ridge_disp = 255 - (ridge_disp * 255 / np.max(ridge))
    ridge_disp = ridge_disp.astype('uint8')

    data.ridge = ridge
    data.ridge_disp = ridge_disp
    p.kernel_size = size
    p.angles = angles
    p.S = S
    p.A = A
    
    autoParameter(data)
    setPanelState(ridge_disp,"Ridge")

def autoParameter(data):
    p = data.parameters
    T = threshold_segments(data.ridge, data.segments)
    p.T = T.copy()
    p.T_new = T.copy()
    segment = int(dropvar.get()[7:]) - 1
    slider_T1.var.set(p.T_new[0, segment])
    slider_T2.var.set(p.T_new[1, segment])

def applyThreshold():
    try:
        global data
        p = data.parameters
    except NameError:
        return

    T_scale = p.T_scale
    if advanced3.get():
        T = p.T_new
    else:
        T = p.T * 1.25**-T_scale

    crackmap = apply_threshold_segments(data.ridge, T, data.segments)
    crackmap_dilated = apply_threshold_segments(data.ridge, T, data.segments, 1)
    overlay = data.im.copy()
    overlay_dilated = data.im.copy()
    color = (255, 0, 0)
    for i in range(3):
        overlay[:,:,i][crackmap!=0] = color[i]
        overlay_dilated[:,:,i][crackmap_dilated!=0] = color[i]

    data.crackmap = crackmap
    data.crackmap_dilated = crackmap_dilated
    data.overlay = overlay
    data.overlay_dilated = overlay_dilated
    data.crackmap_disp = np.logical_and(crackmap,data.mask)
    data.crackmap_dial = np.logical_and(crackmap_dilated,data.mask)
    setPanelState(data.crackmap_disp*255,"CrackMap")

def updatePanel(im, mode=None):
    image = Image.fromarray(np.array(im), mode)
    width, height = image.size
    maxwidth = 1200.0
    maxheight = 750.0
    ratio = 1.0
    if height > maxheight or width > maxwidth:
        ratio = min(maxwidth/width, maxheight/height)
        newsize = (int(width*ratio), int(height*ratio))
        image = image.resize(newsize)
    img2 = ImageTk.PhotoImage(image)
    data.display = img2
    data.disp_ratio = ratio
    panel.configure(image=img2)
    panel.image = img2

def export(im):
    image = Image.fromarray(im)
    ftypes = [('TIFF', '.tif'), ('All files', '*')]
    filename = asksaveasfilename(filetypes=ftypes, defaultextension=".tif")
    image.save(filename)

global data

parameterBoxes = Frame(window, height=2, bd=2)
fileDialogBox = Frame(parameterBoxes, height=2, bd=2)
parameterBox1 = Frame(parameterBoxes, height=2, bd=5)
parameterBox2 = Frame(parameterBoxes, height=2, bd=5)
parameterBox3 = Frame(parameterBoxes, height=2, bd=5)
buttonBox1 = Frame(parameterBoxes, height=2, bd=5)
buttonBox2 = Frame(parameterBoxes, height=2, bd=5)
drawBox = Frame(parameterBoxes, height=2, bd=5)

Label(fileDialogBox, text="File Selection:").pack()
fileName = StringVar()
enterFile = Entry(fileDialogBox, textvariable=fileName).pack(side="left",padx=5)
Button(fileDialogBox, text="Browse", command=setFileName).pack(side="right")
fileDialogBox.pack()

Label(parameterBox1, text="Segment Image by Color (optional)").pack()
sliderFrame1 = Frame(parameterBox1, height=1, bd=1)
slider_clust = SliderObject(sliderFrame1, text="# of colors :", minVal=1, maxVal=16)
slider_resol = SliderObject(sliderFrame1, text="Resolution :", minVal=0.05, maxVal=1, step=0.05)
slider_resol.hide()
sliderFrame1.pack()
Button(parameterBox1, text="Segment", command=segmentImage).pack(side="right")
advanced1 = IntVar()
Checkbutton(parameterBox1, text="Show advanced...", variable=advanced1).pack(side="left")
parameterBox1.pack()


Label(parameterBox2, text="Detect Crack").pack()
radioFrame = Frame(parameterBox2, height=1, bd=1)
crackType = IntVar()
Label(radioFrame, text="Type of crack :", width=11, anchor=E).pack(side="left")
Radiobutton(radioFrame, text="Traditional", variable=crackType, value=0, indicatoron=0, padx=2).pack(side="left")
Radiobutton(radioFrame, text="Vertical", variable=crackType, value=1, indicatoron=0, padx=4).pack(side="left")
Radiobutton(radioFrame, text="Horizontal", variable=crackType, value=2, indicatoron=0, padx=2).pack(side="left")
radioFrame.pack()
sliderFrame2 = Frame(parameterBox2, height=1, bd=1)
slider_csize = SliderObject(sliderFrame2, text="Crack size :", minVal=1, maxVal=8)
slider_csize.var.set(2)
slider_angle = SliderObject(sliderFrame2, text="# of angles :", minVal=1, maxVal=16)
slider_angle.var.set(8)
slider_angle.hide()
slider_S = SliderObject(sliderFrame2, text="S :", minVal=1, maxVal=8)
slider_S.hide()
slider_A = SliderObject(sliderFrame2, text="A :", minVal=1, maxVal=8)
slider_A.hide()
sliderFrame2.pack()
Button(parameterBox2, text="Detect Cracks", command=detectCracks).pack(side="right")
advanced2 = IntVar()
Checkbutton(parameterBox2, text="Show advanced...", variable=advanced2).pack(side="left")
parameterBox2.pack()


Label(parameterBox3, text="Image Threshold").pack()
sliderFrame3 = Frame(parameterBox3, height=1, bd=1)
slider_scale = SliderObject(sliderFrame3, text="Crack density :", minVal=-10, maxVal=10, step=0.01)
slider_T1 = SliderObject(sliderFrame3, text="T1 :", minVal=0.001, maxVal=0.2, step=0.001)
slider_T1.hide()
slider_T2 = SliderObject(sliderFrame3, text="T2 :", minVal=0.001, maxVal=0.2, step=0.001)
slider_T2.hide()
sliderFrame3.pack()
Button(parameterBox3, text="Apply", command=applyThreshold).pack(side="right")
dropvar = StringVar()
dropvar.set('Segment 1')
drop = OptionMenu(parameterBox3, dropvar, "Segment 1")
drop.pack(side="right")
advanced3 = IntVar()
Checkbutton(parameterBox3, text="Show advanced...", variable=advanced3).pack(side="left")
parameterBox3.pack()

Label(drawBox,         text="Editor Tools").pack()
rem = Button(drawBox,  text="Remove Cracks", command=lambda: setBrush(0))
unRem = Button(drawBox,text="UnRemove Cracks", command=lambda: setBrush(1))
rem.pack(              side="left")
unRem.pack(            side="left")
Button(drawBox,        text="Reset Edits", command=lambda:resetMask()).pack(side="left")
drawBox.pack()

Label(buttonBox1,  text="View Images").pack()
Button(buttonBox1, text="Original", command=lambda: setPanelState(data.im,"Image")).pack(side="left")
Button(buttonBox1, text="Segments", command=lambda: setPanelState(data.segments_disp,"SegOverview")).pack(side="left")
Button(buttonBox1, text="Ridge", command=lambda: setPanelState(data.ridge_disp,"Ridge")).pack(side="left")
Button(buttonBox1, text="Crackmap", command=lambda: setPanelState(data.crackmap_disp*255,"CrackMap")).pack(side="left")
Button(buttonBox1, text="Overlay", command=lambda: setPanelState(data.overlay,"OverLay")).pack(side="left")
buttonBox1.pack()

Label(buttonBox2, text="Export Images").pack()
Button(buttonBox2, text="Segments", command=lambda: export(data.segments_disp)).pack(side="left")
Button(buttonBox2, text="Ridge", command=lambda: export(data.ridge_disp)).pack(side="left")
Button(buttonBox2, text="Crackmap", command=lambda: export(data.crackmap_disp)).pack(side="left")
Button(buttonBox2, text="Overlay", command=lambda: export(data.overlay_dilated)).pack(side="left")
buttonBox2.pack()


img = ImageTk.PhotoImage(Image.open("DummyImage.gif"))
panel = Label(window, image=img)

parameterBoxes.pack(side="left")
panel.pack(side="right", fill="both", expand=True)

def newImage(*args):
    im = np.array(Image.open(fileName.get()))
    global data
    data = imageData(im)
    w, h = data.image.size
    ratio = 13 / sqrt(sqrt(w* h))
    ratio = ceil(ratio * 20) / 20
    slider_resol.var.set(ratio)
    setPanelState(im,"Image")

fileName.trace('w', newImage)

def selectSegment(*args):
    try:
        global data
        p = data.parameters
        segments = data.segments
        disp = data.im.copy()
    except NameError:
        return
    segment = int(dropvar.get()[7:]) - 1
    slider_scale.var.set(p.T_scale[segment])
    slider_T1.var.set(p.T_new[0, segment])
    slider_T2.var.set(p.T_new[1, segment])
    try:
        alpha = disp[:,:,3]
    except IndexError:
        alpha = 255*np.ones(disp.shape[:2])
    disp[:,:,:3][segment!=segments] /= 2
    alpha[segment!=segments] /= 2
    disp = np.concatenate((disp[:,:,:3], alpha[:,:,np.newaxis]), 2).astype('uint8')
    setPanelState(disp,"SegSingleView")

dropvar.trace('w', selectSegment)

def setTScale(*args):
    try:
        global data
        p = data.parameters
    except NameError:
        return
    segment = int(dropvar.get()[7:]) - 1
    value = slider_scale.getParameter()
    p.T_scale[segment] = value

def setT1(*args):
    try:
        global data
        p = data.parameters
    except NameError:
        return
    segment = int(dropvar.get()[7:]) - 1
    value = slider_T1.getParameter()
    p.T_new[0, segment] = value

def setT2(*args):
    try:
        global data
        p = data.parameters
    except NameError:
        return
    segment = int(dropvar.get()[7:]) - 1
    value = slider_T2.getParameter()
    p.T_new[1, segment] = value

slider_scale.var.trace('w', setTScale)
slider_T1.var.trace('w', setT1)
slider_T2.var.trace('w', setT2)

def showAdvanced1(*args):
    if advanced1.get():
        slider_resol.show()
    else:
        slider_resol.hide()

def showAdvanced2(*args):
    if advanced2.get():
        slider_csize.hide()
        slider_angle.show()
        slider_S.show()
        slider_A.show()
    else:
        slider_csize.show()
        slider_angle.hide()
        slider_S.hide()
        slider_A.hide()   

def showAdvanced3(*args):
    if advanced3.get():
        slider_scale.hide()
        slider_T1.show()
        slider_T2.show()
    else:
        slider_scale.show()
        slider_T1.hide()
        slider_T2.hide()
        
advanced1.trace('w', showAdvanced1)
advanced2.trace('w', showAdvanced2)
advanced3.trace('w', showAdvanced3)

def changeCrackSize(*args):
    size = slider_csize.getParameter()
    slider_S.var.set(size)
    slider_A.var.set(size)

def changeCrackType(*args):
    if crackType.get():
        slider_angle.disable()
    else:
        slider_angle.enable()
        
slider_csize.var.trace('w', changeCrackSize)
crackType.trace('w', changeCrackType)

def setPanelState(im,panelState):
    updatePanel(im)
    global data
    data.state = panelState
def getPanelState():
    global data
    return data.state
def getPos(event):
    try:
        global data
        disp = data.display
        ratio = data.disp_ratio
    except NameError:
        return
    x, y = event.x, event.y
    x, y = x - panel.winfo_width()/2, y - panel.winfo_height()/2
    x, y = x + disp.width()/2,  y + disp.height()/2
    x, y = int(x / ratio), int(y / ratio)
    return (x,y)
def click(event):
    try:
        global data
        segments = data.segments
    except NameError:
        return
    (x,y) = getPos(event)
    if getPanelState() == "CrackMap" or getPanelState() == "OverLay":
        pass
    else:
        try:
            seg = "Segment " + str(int(segments[y,x] + 1))
            dropvar.set(seg)
        except IndexError:
            return

def draw(event): 
    (x,y) = getPos(event)
    try:
        global data
    except NameError:
        return
    r = data.r
    if getPanelState() == "CrackMap" or getPanelState() == "OverLay":
            for xp in range(x-r,x+r):
                for yp in range(y-r,y+r):
                    try:
                        data.mask[yp][xp] = data.brush
                    except IndexError:
                        return   
            overlay = data.im.copy()
            overlay_dilated = data.im.copy()
            color = (255, 0, 0)
            for i in range(3):
                overlay[:,:,i][data.crackmap_disp!=0] = color[i]
                overlay_dilated[:,:,i][data.crackmap_dial!=0] = color[i]
            data.overlay = overlay
            data.overlay_dilated = overlay_dilated
            data.crackmap_disp = np.logical_and(data.crackmap,data.mask)
            data.crackmap_dial = np.logical_and(data.crackmap_dilated,data.mask)
            if getPanelState() == "CrackMap":
                setPanelState(data.crackmap_disp*200 + data.crackmap*55,"CrackMap")
            elif getPanelState() == "OverLay":
                setPanelState(data.overlay,"OverLay")
def setBrush(a):
    try:
        global data
        data.brush = a
    except NameError:
        return
    if a == 0:
        rem.config(relief = SUNKEN)
        unRem.config(relief = RAISED)
    elif a == 1:
        unRem.config(relief = SUNKEN)
        rem.config(relief = RAISED)
def resetMask():
    try:
        global data
    except NameError:
        return
    data.mask = np.ones(data.im.shape[:2])
    data.crackmap_disp = np.logical_and(data.crackmap,data.mask)
    setPanelState(data.crackmap_disp*255,"CrackMap")

panel.bind('<Button-1>', click)

panel.bind('<B1-Motion>',draw)

window.mainloop()
