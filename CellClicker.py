from asyncio import Lock
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageSequence
from torch._prims_common import apply_perm
from sam2.build_sam import build_sam2 
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from threading import Thread, Lock
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pickle, cv2, csv, time, lzma, shutil, os, copy

class PlotApp:
    def __init__(self, root):
        self.deleterOn = False
        self.selectorOn = False
        self.points = []
        self.bad_points = []
        self.workingImg = None        
        self.finalOutputString = ""
        self.filename = "uninit"
        self.slideNum = 0
        self.epoch = str(time.time())
        # make data dir
        # self.data_dir = C:\\<insert desired path>\\Cell_Clicker_data"
        self.data_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') +  "\\Cell_Clicker_data\\"
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_file_mutex = Lock()
        self.data_filename = self.data_dir + "summary_data_" + self.epoch + ".csv"
        self.root = root
        self.root.title("Object Clicker")
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=10)

        self.current_plot_index = 0
        self.count = 0
        # control frame
        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=0, column=1, padx=0, pady=0, sticky='nsew')        

        # Button to select the next plot
        self.next_plot_button = tk.Button(self.control_frame, text="Next Plot", command=self.next_plot)
        self.next_plot_button.grid(row=0, column=3, padx=5, pady=5)
        
        #button to select the add marker tool
        self.add_marker_button = tk.Button(self.control_frame, text="Add Markers", command=self.selector_button)
        self.add_marker_button.grid(row=0, column=2, padx=5, pady=5)
        
        #button to select the delete marker tool
        self.delete_marker_button = tk.Button(self.control_frame, text="Delete Markers", command=self.deletor_button)
        self.delete_marker_button.grid(row=0, column=1, padx=5, pady=5)

        #set up the counter 
        self.integer_canvas = tk.Canvas(self.control_frame, width=200, height=50, bg='black')
        self.integer_canvas.grid(row=1, column=2, padx=5, pady=5, columnspan=2)

        # browse the filesystem for images
        self.browse_button =tk.Button(self.control_frame, text="Select Image",font=40,command=self.select_file)
        self.browse_button.grid(row=2,column=2)

        # Frame to hold the plot
        self.plot_frame = tk.Frame(root)
        self.plot_frame.grid(row=0, column=0, padx=0, pady=0, sticky='nsew')
        
        self.figure, self.ax = plt.subplots()
        self.figure.set_size_inches(10,10, forward=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        
        # resize the image when resizeing the plot
        self.root.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        self.canvas.draw()

    def select_file(self):
        self.slideNum = 0
        filepath = tk.filedialog.askopenfilename(filetypes=(("tif file", "*.tif"), ("tiff file",'*.tiff'), ("All files", "*.*"),))
        self.workingfile = filepath
        self.filename = filepath.split("/")[-1].split(".")[0] # please don't look, embarasing 
        print(self.filename)
        if filepath.endswith(".tif") or filepath.endswith(".tiff"):
            self.imgs = enumerate(ImageSequence.Iterator(Image.open(filepath)))
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
             img[m] = color_mask
         self.ax.imshow(img)
         
    def on_click(self, event):
        if event.inaxes and self.selectorOn:
            x, y = event.xdata, event.ydata
            self.ax.plot(x, y, 'r+', linewidth=0.5)  # plot a red dot at the click location
            self.points.append((x,y, None)) 
            # appending none here since i do not have the mask for this new point, would have to be processed after the fact in some way 
            self.count = self.count + 1
        elif event.inaxes and self.deleterOn:
            x,y = event.xdata, event.ydata
            closest_point = None
            min_distance = float('inf')
            for px,py,obj in self.points:
                distance = np.sqrt((x - px)**2 + (y - py)**2)
                if distance < min_distance:
                    closest_point = (px,py, obj)
                    min_distance = distance
            if closest_point:
                self.bad_points.append(closest_point)
                self.points.remove(closest_point)
                self.count -= 1
                #TODO can't be the most efficient way to do this
                self.ax.clear()
                self.ax.axis('off')
                self.ax.imshow(self.workingImg)
                pts = []
                for x,y,z in self.points:
                    point = self.ax.plot(x, y, 'r+', linewidth=0.5)
                    pts.append((x,y, z))
                self.points = pts
        self.canvas.draw()
        self.integer_canvas.delete("all")
        self.integer_canvas.create_text(180, 20, text=str(self.count), fill="red", font=("Arial", 24), anchor="e")        
                
                


    def segment_next(self, img):
        try:
            image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        except StopIteration:
            return 0, []
        sam2_checkpoint = "sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False) # todo test apply_postprocessing=True
        mask_generator = SAM2AutomaticMaskGenerator(sam2, crop_n_layers=1)
        masks = mask_generator.generate(image)
        print(len(masks))
        print(masks[0].keys())
        self.ax.axis('off')
        self.ax.imshow(img, aspect='equal')
        # sort to remove to biggest mask
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sorted_masks = sorted_masks[1:]
        pts = []
        for mask in sorted_masks:
            x, y, w, h = mask['bbox']
            self.ax.plot(x + w/2, y + h/2, 'r+', linewidth=0.5)
            pts.append((x + w/2, y + h/2, mask))
        return len(masks), pts
    
    def compress_to_lzma(self, file_obj, output_lzma_path):
        with lzma.open(output_lzma_path, 'wb') as f_out:
            shutil.copyfileobj(file_obj, f_out)

    def pack_data(self, _filename, _slideNum, _points, _bad_points, _count):
        # pack and write csv with, filename, total, good point valuse, serialize the mask data for future use
        # the serialized comes out with size on the order of GB, so we will compress the data, its seems to have a VERY high compression rate
        # copy.deepcopy(self.filename), copy.deepcopy(self.slideNum), copy.deepcopy(self.points), copy.deepcopy(self.bad_points),copy.deepcopy(self.count
   #     _filename   = data_list[0]
   #        = data_list[1]
   #          = data_list[2]
   #      = data_list[3]
   #           = data_list[4]
        # self.data_dir
        goodfilename = self.data_dir + "good_points_serialization" + self.epoch + "_" + _filename + "_slide" + str(_slideNum) 
        badfilename =  self.data_dir + "bad_points_serialization" + self.epoch + "_" + _filename + "_slide" + str(_slideNum)
        goodfile  = open(goodfilename + ".bin", 'wb')
        badfile  = open(badfilename + ".bin", 'wb')
        pickle.dump(_points, goodfile)
        pickle.dump(_bad_points, badfile)
        goodfile.close()
        badfile.close()
        goodfileR  = open(goodfilename + ".bin", 'rb')
        badfileR  = open(badfilename + ".bin", 'rb')
        self.compress_to_lzma(goodfileR, goodfilename + ".xz")
        self.compress_to_lzma(badfileR,  badfilename + ".xz")
        goodfileR.close()
        badfileR.close()
        # remove temp files
        os.remove(goodfilename + ".bin")            
        os.remove(badfilename + ".bin")            
        goodpoints = list(map(lambda x: (x[0], x[1]), _points))
        badpoints  =  list(map(lambda y: (y[0], y[1]), _bad_points))
        with self.data_file_mutex:
            with open(self.data_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.workingfile, _slideNum ]) # info row
                writer.writerow([str(_count)]) # count row ,, is wack
                writer.writerow("GOOD POINTS") # good points
                writer.writerow(goodpoints) # good points
                writer.writerow("BAD POINTS")  
                writer.writerow(badpoints)  

    def next_plot(self):
        self.deleterOn = False
        self.selectorOn = False
        if self.count != 0:
            self.finalOutputString = str(self.count) + ","  + self.finalOutputString
            print(f"summary of counts {self.finalOutputString}")
        # save off old data for the pack_data to zip up in another thread
        # filename , slideNum, points, bad_points, count 
        data_copy = [copy.deepcopy(self.filename), copy.deepcopy(self.slideNum), copy.deepcopy(self.points), copy.deepcopy(self.bad_points),copy.deepcopy(self.count)]
        thread = Thread(target=self.pack_data, args=(data_copy))
        thread.start()
        time.sleep(1) # cause why not
                
        self.slideNum = self.slideNum + 1
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
