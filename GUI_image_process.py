from math import *
from tkinter import *
import cv2 as cv
from PIL import Image,ImageTk
import numpy as np
import tkinter.filedialog as filedialog
import tkinter.simpledialog as simpledialog

class window():
    root:Tk
    # Intial Image
    im_src:np.array = None
    # Store image for Processing
    im_dist:list = []
    # Display Image, will be changed later
    im_show:np.array = None
    im_tk:PhotoImage
    menubar:Menu
    imageforshow:Canvas
    file_path_open:str
    width:int
    height:int

    def __init__(self):
        self.im_src = cv.imread("./src/Altair_dagger.jpg", cv.COLOR_BGR2GRAY)
        self.initial()
        self.root.mainloop()
    # Initialization
    def initial(self):
        self.root = Tk()
        self.root.title("Digital Image Processing Toolbox")
        self.width_max,self.height_max=self.root.maxsize()
        self.root.geometry("%sx%s"%(int(self.width_max/2),int(self.height_max/2)))
        self.initial_menu()
        # self.imLable.pack()
        self.imageforshow = Canvas(self.root)
        self.imageforshow.pack()
        self.root.config(menu=self.menubar)
    # Initialize the menu button and bind the function that handles the click time
    def initial_menu(self):
        self.menubar = Menu(self.root)
        #Create File Submenu
        self.menu_file=Menu(self.menubar)
        self.menu_file.add_command(label="Load Image",command=self.openDir)
        self.menu_file.add_command(label="Save",command=self.saveDir)
        self.menu_file.add_command(label="Undo",command=self.back)
        self.menubar.add_cascade(label="File",menu=self.menu_file)

        #Create Translation Submenu
        self.menu_translate=Menu(self.menubar)
        self.menu_translate.add_command(label="Translate in x",command=self.translate_x)
        self.menu_translate.add_command(label="Translate in y",command=self.translate_y)
        self.menubar.add_cascade(label="Translation",menu=self.menu_translate)

        #Create Transformation Submenu
        self.menu_transform=Menu(self.menubar)
        self.menu_transform.add_command(label="Skewing",command=self.skewing)
        self.menu_transform.add_command(label="Flipping",command=self.flipping)
        self.menu_transform.add_command(label="Blending",command=self.blending)
        self.menubar.add_cascade(label="Transformation",menu=self.menu_transform)

        #Create Rotate & Scale Submenu
        self.menu_RandS=Menu(self.menubar)
        self.menu_RandS.add_command(label="Rotate",command=self.rotate)
        self.menu_RandS.add_command(label="Scale",command=self.scale)
        self.menubar.add_cascade(label="Rotate & Scale",menu=self.menu_RandS)
        #Create Grayscale Submenu
        self.menu_gray=Menu(self.menubar)
        self.menu_gray.add_command(label="Convert to Grayscale",command=self.rgb2gray)
        self.menu_gray.add_command(label="Binary Graph",command=self.threshold)
        self.menu_gray.add_cascade(label="OTSU",command=self.OTSU)
        self.menubar.add_cascade(label="Grayscale",menu=self.menu_gray)
        #Create Image Smoothing Submenu
        self.menu_noiseremove=Menu(self.menubar)
        self.menu_noiseremove.add_command(label="Mean Filter",command=self.mean_remove)
        self.menu_noiseremove.add_command(label="Median Filter",command=self.median_remove)
        self.menu_noiseremove.add_command(label="Gaussian Smoothing",command=self.gauss_remove)
        self.menubar.add_cascade(label="Image Smoothing",menu=self.menu_noiseremove)
        #Create Image Enhancement Submenu
        self.menu_inhance=Menu(self.menubar)
        self.menu_inhance.add_command(label="Histogram Equalization",command=self.hist_equalize)
        self.menubar.add_cascade(label="Image Enhancement",menu=self.menu_inhance)
        #Create Image Sharpening Submenu
        self.submenu_sharper=Menu(self.menu_inhance)
        self.submenu_sharper.add_command(label="Sobel",command=self.inhance_sobel)
        self.submenu_sharper.add_command(label="Laplacian",command=self.inhance_laplacian)
        self.submenu_sharper.add_command(label="Canny",command=self.inhance_Canny)
        self.menu_inhance.add_cascade(label="Image Sharpening",menu=self.submenu_sharper)
        #Create Grayscale Transformation Submenu
        self.submenu_tranlation=Menu(self.menu_inhance)
        self.submenu_tranlation.add_command(label="Image Inversion",command=self.translation_reverse)
        self.submenu_tranlation.add_command(label="Logarithmic Transformation",command=self.translation_log)
        self.submenu_tranlation.add_command(label="Gamma Transform",command=self.translation_gamma)
        self.menu_inhance.add_cascade(label="Grayscale Transformation",menu=self.submenu_tranlation)
        #Create Morphological Transformation Submenu
        self.menu_morphological=Menu(self.menubar)
        self.menu_morphological.add_command(label="Corrosion",command=self.morphology_erode)
        self.menu_morphological.add_command(label="Swell",command=self.morphology_dailate)
        self.menu_morphological.add_command(label="Open Operation",command=self.morphology_open)
        self.menu_morphological.add_command(label="Close Operation",command=self.morphology_close)
        self.menubar.add_cascade(label="Morphological Transformation",menu=self.menu_morphological)
        # Set Right Step Display Area
        self.right_area=Frame(self.root)
        self.right_area.pack(side=RIGHT,fill=Y)
        self.list_procedure = Listbox(self.right_area)
        self.scollbar=Scrollbar(self.right_area)
        self.scollbar.pack(side=RIGHT,fill=Y)
        self.scollbar.config(command=self.list_procedure.yview)
        self.list_procedure.config(yscrollcommand=self.scollbar.set,height=30)
        self.list_procedure.pack(fill=Y)
        self.botton_see=Button(self.right_area,command=self.review,text="Check").pack()

# Click Response
    # Open Image Click Response
    def openDir(self):
        self.delet_all()
        files = [("PNG", "*.png"), ("JPG(JPEG)", "*.j[e]{0,1}pg"), ("All Files", "*")]
        self.file_path_open = filedialog.askopenfilename(title="Select Image", filetypes=files)
        if len(self.file_path_open) != 0:
            self.im_src = window.cv_imread(self.file_path_open)
            self.im_src = cv.cvtColor(self.im_src, cv.COLOR_RGB2GRAY)
            self.show_image(self.im_src)
            self.finish_process(self.im_src.copy(),"read in a image")
    def saveDir(self):
        files = [("PNG", ".png"), ("JPG(JPEG)", ".j[e]{0,1}pg")]
        self.file_path_save = filedialog.asksaveasfilename(title="Choose a save Path",initialfile= "", defaultextension=".png",filetypes=files)
        if len(self.file_path_save) != 0:
            im = self.pop()
            if im is not None:
                cv.imwrite(self.file_path_save,im)
                simpledialog.messagebox.showinfo("Successfully Saved","path"+self.file_path_save)
            else:
                simpledialog.messagebox.showerror("No Picture has been Selected")
    # Image Smoothing-Mean Filter
    def mean_remove(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("Size of the convolution kernel for the input smoothing operation", "", initialvalue=3, minvalue=1, maxvalue=msize)
        if ksize is None:
            return
        if ksize%2 == 0:
            simpledialog.messagebox.showerror("Parameter Error","Kernel Size should be odd")
            return
        kernel = np.ones((ksize,ksize), np.uint8)
        kernel = kernel/ksize/ksize
        im = cv.filter2D(im_dist,-1,kernel)
        self.finish_process(im,"mean filer")
    # Image smoothing-Median Filter
    def median_remove(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        msize = min(im_dist.shape[:2])
        ksize=None
        ksize = simpledialog.askinteger("Kernel size of the input Smoothing Operation", "", initialvalue=3, minvalue=1, maxvalue=msize)
        if ksize is None:
            return
        if ksize%2 == 0:
            simpledialog.messagebox.showerror("Parameter Error","Kernel size should be odd")
            return
        im = cv.medianBlur(im_dist,ksize)
        self.finish_process(im, "median filer")
    # Image smoothing-Gaussian Filter
    def gauss_remove(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("Kernel size of the input Smoothing Operation", "", initialvalue=3, minvalue=1, maxvalue=msize)
        if ksize is None:
            return
        if ksize % 2 == 0:
            simpledialog.messagebox.showerror("Parameter Error", "Kernel size should be odd")
            return
        sigmaX = simpledialog.askfloat("Set Gaussian filter parameter sigmaX","",initialvalue=1, minvalue=1, maxvalue=msize)
        if sigmaX is None:
            return
        sigmaY = simpledialog.askfloat("Set Gaussian filter parameter sigmaY","",initialvalue=sigmaX, minvalue=1, maxvalue=msize)
        if ksize is None:
            return

        im = cv.GaussianBlur(im_dist,(ksize,ksize),sigmaX=sigmaX,sigmaY=sigmaY)
        self.finish_process(im, "gauss filer")
    # Morphological Operations-Corrosion
    def morphology_erode(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            self.rgb2gray()
            im_dist = cv.cvtColor(im_dist, cv.COLOR_RGB2GRAY)
        if np.logical_and(im_dist>0,im_dist<255).any():
            #Binarization is required
            self.OTSU()
            _,im_dist = cv.threshold(im_dist,0,255,cv.THRESH_OTSU)
        msize=min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("Kernel Size of the Input Erosion Operation","",initialvalue=3,minvalue=1,maxvalue=msize)
        if ksize is None:
            return
        time = simpledialog.askinteger("Enter the number of operations","",initialvalue=1,minvalue=1)
        if time is None:
            return
        ksize = (ksize,ksize)
        kernel=np.ones(ksize,np.uint8)
        im = cv.morphologyEx(im_dist,cv.MORPH_ERODE,kernel=kernel,iterations=time)
        self.finish_process(im,"erode x"+str(time)+" ksize:"+str(ksize))
    # Morphological Operations-Swell
    def morphology_dailate(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            self.rgb2gray()
            im_dist = cv.cvtColor(im_dist, cv.COLOR_RGB2GRAY)
        if np.logical_and(im_dist > 0, im_dist < 255).any():
            # Binarization is Required
            self.OTSU()
            _, im_dist = cv.threshold(im_dist, 0, 255, cv.THRESH_OTSU)
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("Kernel Size of the Input Dilation Operation", "", initialvalue=3,minvalue=1,maxvalue=msize)
        if ksize is None:
            return
        time = simpledialog.askinteger("Enter the number of operations", "", initialvalue=1, minvalue=1)
        if time is None:
            return
        ksize = (ksize, ksize)
        kernel = np.ones(ksize, np.uint8)
        im = cv.morphologyEx(im_dist, cv.MORPH_DILATE, kernel=kernel, iterations=time)
        self.finish_process(im, "dailate x" + str(time) + " ksize:" + str(ksize))
    #Morphological Operations-Open
    def morphology_open(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            self.rgb2gray()
            im_dist = cv.cvtColor(im_dist, cv.COLOR_RGB2GRAY)
        if np.logical_and(im_dist > 0, im_dist < 255).any():
            # Binarization is Required
            self.OTSU()
            _, im_dist = cv.threshold(im_dist, 0, 255, cv.THRESH_OTSU)
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("Enter the convolution kernel size of the open operation", "", initialvalue=3,minvalue=1,maxvalue=msize)
        if ksize is None:
            return
        time = simpledialog.askinteger("Enter the number of operations", "", initialvalue=1, minvalue=1)
        if time is None:
            return
        ksize = (ksize, ksize)
        kernel = np.ones(ksize, np.uint8)
        im = cv.morphologyEx(im_dist, cv.MORPH_OPEN, kernel=kernel, iterations=time)
        self.finish_process(im, "open x" + str(time) + " ksize:" + str(ksize))
    # Morphological Operations-Close
    def morphology_close(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            self.rgb2gray()
            im_dist = cv.cvtColor(im_dist, cv.COLOR_RGB2GRAY)
        if np.logical_and(im_dist > 0, im_dist < 255).any():
            # Binarization is required
            self.OTSU()
            _, im_dist = cv.threshold(im_dist, 0, 255, cv.THRESH_OTSU)
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("Size of the convolution kernel for the input closing operation", "", initialvalue=3, minvalue=1, maxvalue=msize)
        if ksize is None:
            return
        time = simpledialog.askinteger("Enter the number of operations", "", initialvalue=1, minvalue=1)
        if time is None:
            return
        ksize = (ksize, ksize)
        kernel = np.ones(ksize, np.uint8)
        im = cv.morphologyEx(im_dist, cv.MORPH_CLOSE, kernel=kernel, iterations=time)
        self.finish_process(im, "close x" + str(time) + " ksize:" + str(ksize))
    # Sobel Operation
    def inhance_sobel(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            self.rgb2gray()
            im_dist=cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)

        x = cv.Sobel(im_dist, cv.CV_16S, 1, 0)
        y = cv.Sobel(im_dist, cv.CV_16S, 0, 1)

        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)

        dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        dst = np.uint8(dst)
        self.finish_process(dst,"sobel edge")

    # Laplacian Operation
    def inhance_laplacian(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            self.rgb2gray()
            im_dist=cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)
        dst = cv.Laplacian(im_dist,cv.CV_16S)
        dst = np.uint8(dst)
        self.finish_process(dst,"laplacian edge")


    # Canny Operation
    def inhance_Canny(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            self.rgb2gray()
            im_dist=cv.cvtColor(im_dist,cv.COLOR_BGR2GRAY)
        dst = cv.Canny(im_dist,125,255)
        dst = np.uint8(dst)
        self.finish_process(dst, "canny edge")

    # Image Inversion
    def translation_reverse(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        if len(im_dist) == 3:
            b,g,r=cv.split(im_dist)
            r = 255-r
            g = 255-g
            b = 255-b
            im_dist=cv.merge([b,g,r])
        else:
            im_dist = 255-im_dist
        self.finish_process(im_dist,"reverse image")
    # Gamma Transform
    def translation_gamma(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        c = simpledialog.askfloat("Set the parameter c of the gamma transform","s=cr^γ")
        if c is None:
            return
        gamma = simpledialog.askfloat("Set the parameter γ of the gamma transform","s=cr^γ")
        if gamma is None:
            return
        if len(im_dist) == 3:
            for k in range(3):
                im = im_dist[:,:,k]
                max_pixel=np.max(im)
                im=np.uint8(
                    c*np.power(im/max_pixel,gamma)*max_pixel
                )
                im_dist[:,:,k]=im
        else:
            max_pixel = np.max(im_dist)
            im_dist = np.uint8(
                    c * np.power(im_dist / max_pixel, gamma) * max_pixel
            )
        self.finish_process(im_dist,"gamma with coefficient c="+str(c)+" gamma="+str(gamma))
    # Logarithmic Transformation
    def translation_log(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        c = simpledialog.askfloat("Set the parameter c for the logarithmic transformation","s = c*log(1+r)")
        if c is None:
            return
        if len(im_dist) == 3:
            for k in range(3):
                im = im_dist[:,:,k]
                max_pixel=np.max(im)
                im=np.uint8(
                    (c*np.log(1+im)-c*np.log(1+0))/\
                        (c*np.log(1+max_pixel)-c*np.log(1+0))*max_pixel
                            )
                im_dist[:,:,k]=im
        else:
            max_pixel = np.max(im_dist)
            im_dist = np.uint8(
                    (c * np.log(1 + im_dist) - c * np.log(1 + 0)) / \
                      (c * np.log(1 + max_pixel) - c * np.log(1 + 0)) * max_pixel
            )
        self.finish_process(im_dist,"log with coefficient "+str(c))

    # Rotate Operation
    def rotate(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        (h, w) = im_dist.shape[:2]
        degree = simpledialog.askfloat("Enter rotation angle",
                              "Clockwise is negative, Counterclockwise is positive",
                              initialvalue=90,
                              maxvalue=180,
                              minvalue=-180)
        if degree is None:
            return
        hNew = int(w * fabs(sin(radians(degree))) + h * fabs(cos(radians(degree))))
        wNew = int(h * fabs(sin(radians(degree))) + w * fabs(cos(radians(degree))))
        center = (w//2,h//2)

        M = cv.getRotationMatrix2D(center, degree, 1.0)
        M[0, 2] += (wNew - w) / 2
        M[1, 2] += (hNew - h) / 2
        im = cv.warpAffine(im_dist, M, ( wNew,hNew), borderValue=(255, 255, 255))
        self.finish_process(im,"rotate "+str(degree)+"degree")
    # Translate in x direction
    def translate_x(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        (h, w) = im_dist.shape[:2]
        rate = simpledialog.askfloat("Enter the translation in pixels",
                                     "To the right is positive, to the left is negative",
                                     initialvalue=10,)
        if rate is None:
            return
        T = np.float32([[1, 0, rate], [0, 1, 1]])
  
        # We use warpAffine to transform
        # the image using the matrix, T
        im_dist = cv.warpAffine(im_dist, T, (w, h))
        self.finish_process(im_dist,"translate image by "+str(rate)+" pixels")
    # Translate in y direction
    def translate_y(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        (h, w) = im_dist.shape[:2]
        rate = simpledialog.askfloat("Enter the translation in pixels",
                                     "Down is positive, up is negative",
                                     initialvalue=10,)
        if rate is None:
            return
        T = np.float32([[1, 0, 1], [0, 1, rate]])
  
        # We use warpAffine to transform
        # the image using the matrix, T
        im_dist = cv.warpAffine(im_dist, T, (w, h))
        self.finish_process(im_dist,"translate image by "+str(rate)+" pixels")
    # Skew Image
    def skewing(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        (h, w) = im_dist.shape[:2]
        rate = simpledialog.askfloat("Enter the skewing in pixels",
                                     "To the right is positive, to the left is negative",
                                     initialvalue=50,)
        if rate is None:
            return
        pts1 = np.float32(
            [[0, 0],
            [h-1, 0],
            [0, w-1]]
        )
        pts2 = np.float32(
            [[0, 0],
            [h-1, 0],
            [rate, w-1]]
        )    
        M = cv.getAffineTransform(pts1,pts2)
        im_dist = cv.warpAffine(im_dist, M, (w, h))
        self.finish_process(im_dist,"skew image by "+str(rate)+" pixels")
    # Flipping Image
    def flipping(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        (h, w) = im_dist.shape[:2]
        direction = simpledialog.askfloat("Enter the flipping direction",
                                     "0 vertically (around x-axis), 1 horizontally (around y-axis), -1 (around xy-axis)",
                                     initialvalue=50,)
        if direction is None:
            return

        im_dist = cv.flip(im_dist, int(direction))
        if direction == 1:
            self.finish_process(im_dist,"flip image around y axis")
        elif direction == 0:
            self.finish_process(im_dist,"flip image around x axis")
        elif direction == -1:
            self.finish_process(im_dist,"flip image around xy axis")
    # Blending two Image
    def blending(self):
        im_dist=self.pop()
        files = [("PNG", "*.png"), ("JPG(JPEG)", "*.j[e]{0,1}pg"), ("All Files", "*")]
        self.file_path_open = filedialog.askopenfilename(title="Select Image", filetypes=files)
        if len(self.file_path_open) != 0:
            second_image = cv.imread(self.file_path_open)
            second_image = cv.cvtColor(second_image, cv.COLOR_RGB2GRAY)
            self.finish_process(im_dist,"read in a image")
        
        if im_dist is None:
            return
        if second_image is None:
            return

        (h, w) = im_dist.shape[:2]
        second_image = cv.resize(second_image, (w, h), interpolation = cv.INTER_AREA)
        alpha = 0.5
        input_alpha = simpledialog.askfloat("Enter the blending percentage",
                                     "value vary between 0.0 and 1.0",
                                     initialvalue=0.5,)
        if 0 <= alpha <= 1:
            alpha = float(input_alpha)
        beta = (1.0 - alpha)
        im_dist = cv.addWeighted(im_dist, alpha, second_image, beta, 0.0)
        self.finish_process(im_dist,"blending two images")
    # Scaling Operation
    def scale(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        (h, w) = im_dist.shape[:2]
        rate = simpledialog.askfloat("Enter the zoom ratio",
                                     "This scaling is proportional scaling. After the scaling operation, if it exceeds the screen display range, the display size will be adjusted to an appropriate range, and the size of the original image has been changed according to the scaling ratio.",
                                     initialvalue=1,
                                     minvalue=0.1)
        if rate is None:
            return
        im_dist = cv.resize(im_dist,dsize=(0,0),fx=rate,fy=rate)
        self.finish_process(im_dist,"scale to initial's "+str(rate))
    # Histogram Equalization Operation
    def hist_equalize(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        if im_dist.shape[-1] == 3:
            r,g,b=cv.split(im_dist)
            r = cv.equalizeHist(r)
            g = cv.equalizeHist(g)
            b = cv.equalizeHist(b)
            im_dist = cv.merge((r,g,b))
        else:
            im_dist = cv.equalizeHist(im_dist)
        self.finish_process(im_dist,"hist_equalize")
    # Grayscale Operation
    def rgb2gray(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            im_dist = cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)
            self.finish_process(im_dist,"RGB to GRAY")
        else:
            self.finish_process(im_dist,"GRAY to GRAY")
    # Binarization Operation
    def threshold(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        thresh_min = simpledialog.askinteger("Enter the minimum threshold",
                                "Pixel values below this value will be set to 0",
                                initialvalue=125,
                                maxvalue=255,
                                minvalue=1)
        if thresh_min is None:
            return
        thresh_max = simpledialog.askinteger("Enter the maximum threshold",
                                             "Pixel values above this value will be set to 0",
                                             initialvalue = 255,
                                             maxvalue=255,
                                             minvalue=1)
        if thresh_max is None:
            return
        if thresh_max <= thresh_min:
            simpledialog.messagebox.showerror("The max threshold needs to be greater than the max threshold")
            return
        if len(im_dist.shape) == 3:
            im_dist = cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)
            self.rgb2gray()
        _,im_dist=cv.threshold(im_dist,thresh_min,thresh_max,cv.THRESH_BINARY)
        self.finish_process(im_dist,"thresh "+str(thresh_min)+" to "+str(thresh_max))
    # Optimal Threshold Segmentation
    def OTSU(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) == 3:
            im_dist = cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)
            self.rgb2gray()
        thresh,im_dist=cv.threshold(im_dist,0,255,cv.THRESH_OTSU)
        self.finish_process(im_dist,"thresh OTSU "+str(thresh))
# Tool Method
    # Post-Image Maintenance Operation
    def finish_process(self,im,name:str):
        self.show_image(im)#Display Image
        self.im_dist.append(im)#Cache Image
        self.list_procedure.insert(END,name)#Cache Steps
        self.root.update()#Update

    # Get the cached image location that the user wants to view, and display the corresponding image
    def review(self):
        indexs=self.list_procedure.curselection()
        self.show_image(self.im_dist[indexs[0]])

    # Image Stack Pop
    def pop(self,delete=False):
        if len(self.im_dist) == 0:
            return None
        if delete == False:
            # Used during image processing, only to get a copy of the latest processed image
            return self.im_dist[-1].copy()
        else:
            # Used when calling back to the previous step to return the image and delete the cache
            self.list_procedure.delete(END)
            return self.im_dist.pop()
    # Display Image
    def show_image(self,im,newsize:tuple=None):
        self.im_show = im.copy()
        if newsize is None:
            (imsizecol,imsizerow) = self.get_reshape_size(self.im_show)
        else:
            (imsizecol,imsizerow) = newsize
        self.im_show = cv.resize(self.im_show, (imsizecol, imsizerow))
        self.root.geometry("%sx%s"%(int(self.width_max*4/5),int(self.height_max*4/5)))
        self.im_tk = window.im_np2im_tk(self.im_show)
        self.imageforshow.config(height = imsizerow,width=imsizecol)
        self.imageforshow.create_image(0,0,anchor=NW,image=self.im_tk)
        self.root.update()

    # Returns the last processed image and deletes the cache
    def back(self):
        self.pop(delete=True)
        self.show_image(self.pop())

    # Get the size of the image that will best display it
    def get_reshape_size(self,im):
        if len(im.shape) == 3:
            imsizerow, imsizecol, _ = im.shape
        else:
            imsizerow, imsizecol = im.shape

        #Resize the image to the best size
        if imsizecol > self.width_max*0.7 or imsizerow > self.height_max*0.7:
            if imsizecol > imsizerow:
                imsizerow = int(self.width_max*0.7*imsizerow/imsizecol)
                imsizecol = int(self.width_max*0.7)

            if imsizecol <= imsizerow:
                imsizecol = int(self.height_max*0.7*imsizecol/imsizerow)
                imsizerow = int(self.height_max*0.7)

        return (imsizecol,imsizerow)

    def im_np2im_tk(im):
        # Change the order of the three-channel arrangement and convert the image to a displayable type
        if len(im.shape) == 3:
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        img = Image.fromarray(im)
        imTk = ImageTk.PhotoImage(image=img)
        return imTk

    def delet_all(self):
        if len(self.im_dist) != 0:
            ok = simpledialog.messagebox.askokcancel("Note","Opening a new picture will delete all existing pictures, are you sure you want to continue opening")
            if ok is None:
                return
            if ok is False:
                return
            self.im_dist.clear()
            self.list_procedure.delete(0,END)

    def cv_imread(file_path=""):
        #Encoding format conversion
        cv_img = cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
        return cv_img

w = window()