from os import close
import tkinter as tk
from tkinter import ttk
from turtle import distance
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageSequence
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class PlotApp:
    def __init__(self, root):
        self.deleterOn = False
        self.selectorOn = False
        self.points = []
        self.workingImg = None        
        self.finalOutputString = ""
        
        self.root = root
        # self.root.attributes("-fullscreen",True)
        self.root.title("Object Clicker")
        # Configure grid to scale with window
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=10)
        #self.root.state('zoomed')

        self.current_plot_index = 0
        self.count = 0
        # control frame
        self.control_frame = ttk.Frame(root)
        self.control_frame.grid(row=0, column=1, padx=0, pady=0, sticky='nsew')        

        # Button to select the next plot
        self.next_plot_button = ttk.Button(self.control_frame, text="Next Plot", command=self.next_plot)
        self.next_plot_button.grid(row=0, column=3, padx=5, pady=5)
        
        #button to select the add marker tool
        self.add_marker_button = ttk.Button(self.control_frame, text="Add Markers", command=self.selector_button)
        self.add_marker_button.grid(row=0, column=2, padx=5, pady=5)
        
        #button to select the delete marker tool
        self.delete_marker_button = ttk.Button(self.control_frame, text="Delete Markers", command=self.deletor_button)
        self.delete_marker_button.grid(row=0, column=1, padx=5, pady=5)

        #set up the counter 
        self.integer_canvas = tk.Canvas(self.control_frame, width=200, height=50, bg='black')
        self.integer_canvas.grid(row=1, column=2, padx=5, pady=5, columnspan=2)

        # browse the filesystem for images
        self.browse_button =tk.Button(self.control_frame, text="Select Image",font=40,command=self.select_file)
        self.browse_button.grid(row=2,column=2)

        # Frame to hold the plot
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.grid(row=0, column=0, padx=0, pady=0, sticky='nsew')
        
        self.figure, self.ax = plt.subplots()
        self.figure.set_size_inches(10,10, forward=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        
        # resize the image when resizeing the plot
        self.root.bind("<Configure>", self.on_resize)
        #self.next_plot()

    def on_resize(self, event):
        self.canvas.draw()

    def select_file(self):
        filename = tk.filedialog.askopenfilename(filetypes=(("tif file", "*.tif"), ("tiff file",'*.tiff'), ("All files", "*.*"),))
        self.workingfile = filename
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            self.imgs = enumerate(ImageSequence.Iterator(Image.open(filename)))
            self.next_plot()
        # update image?

    def show_anns(self, anns):
         print("superceeded, kept for debugging purposes") 
         if len(anns) == 0:
             return
         sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
         # jank way to remove the background mask
         sorted_anns = sorted_anns[1:]
         self.ax = plt.gca()
         self.ax.set_autoscale_on(False)
    
         img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
         img[:,:,3] = 0
         for ann in sorted_anns:
             m = ann['segmentation']
             color_mask = np.concatenate([np.random.random(3), [0.35]])
             #color_mask = np.concatenate([(255,0,0), [0.35]])
             img[m] = color_mask
         self.ax.imshow(img)

    def on_click(self, event):
        if event.inaxes and self.selectorOn:
            x, y = event.xdata, event.ydata
            self.ax.plot(x, y, 'r+', linewidth=0.5)  # plot a red dot at the click location
            self.points.append((x,y))
            self.count = self.count + 1
        elif event.inaxes and self.deleterOn:
            x,y = event.xdata, event.ydata
            closest_point = None
            min_distance = float('inf')
            for px,py in self.points:
                distance = np.sqrt((x - px)**2 + (y - py)**2)
                if distance < min_distance:
                    closest_point = (px,py)
                    min_distance = distance
            if closest_point:
                self.points.remove(closest_point)
                self.count -= 1
                #TODO can't be the most efficient way to do this
                self.ax.clear()
                self.ax.axis('off')
                self.ax.imshow(self.workingImg)
                pts = []
                for x,y in self.points:
                    point = self.ax.plot(x, y, 'r+', linewidth=0.5)
                    pts.append((x,y))
                self.points = pts
        self.canvas.draw()
        self.integer_canvas.delete("all")
        self.integer_canvas.create_text(180, 20, text=str(self.count), fill="red", font=("Arial", 24), anchor="e")        
                
                


    def segment_next(self, img):
        image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        #default args 
        #mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95,stability_score_offset=1.0, box_nms_thresh=0.7, crop_n_layers=0)
        mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95,stability_score_offset=1.0, box_nms_thresh=0.7, crop_n_layers=1)
        masks = mask_generator.generate(image)
        print(len(masks))
        print(masks[0].keys())
        self.ax.axis('off')
        self.ax.imshow(img, aspect='equal')
        # sort to remove to big mask
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sorted_masks = sorted_masks[1:]
        pts = []
        for mask in sorted_masks:
            x, y, w, h = mask['bbox']
            self.ax.plot(x + w/2, y + h/2, 'r+', linewidth=0.5)
            pts.append((x + w/2, y + h/2))
        return len(masks), pts

    def next_plot(self):
        self.deleterOn = False
        self.selectorOn = False
        if self.count != 0:
            self.finalOutputString = str(self.count) + ","  + self.finalOutputString
            print(f"summary of counts {self.finalOutputString}")
            
        plot_type = "Cells"
        self.ax.clear()
        i, img = next(self.imgs)
        self.workingImg = img
        self.count, self.points = self.segment_next(img)
        self.current_plot_index = self.current_plot_index + 1 
        self.integer_canvas.delete("all")
        self.integer_canvas.create_text(180, 20, text=str(self.count), fill="red", font=("Arial", 24), anchor="e")
        self.canvas.draw()
        
    def selector_button(self):
        #unset deletor # TODO add something to tell the user if the selector/deletor is on
        self.deleterOn = False
        if self.selectorOn:
            self.selectorOn = False
        else:
            self.selectorOn = True
            # bind the selector
            
        
    def deletor_button(self):
        #unset selector # TODO add something to tell the user if the selector/deletor is on
        self.selectorOn = False
        if self.deleterOn:
            self.deleterOn = False
        else:
            self.deleterOn = True


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()