import sys
import cv2
import tkinter as tk
from PIL import ImageTk
from PIL import Image

#COMMENT
class PixelSelect(tk.Frame):

	"""
	Frame that fills the window Tk(). Container for the two frames; 
	'ImageFrame' and 'BottomFrame'

	Args:
		imgPath(String)	: The path of the image
	"""

	def __init__(self, imgPath):
		self.result = 0
		self.numSegments = 0
		self.imgPath = imgPath


	def startGUI(self):

		"""
		Initialize GUI for marker point select for use in Watershed segmentation algorithm

		Marker points can be found in self.result after GUI has finished
		"""

		self.root = tk.Tk()
		tk.Frame.__init__(self, self.root)
		self.grid()

		self.imFrame = ImageFrame(self, self.imgPath)
		self.bottomFrame = BottomFrame(self)

		self.imFrame.grid(row=0,column=0)
		self.bottomFrame.grid(row=1,column=0)

		self.root.mainloop()

	def exitGUI(self):

		"""
		Close current session of GUI, storing marker points in self.result
		"""

		self.result = self.imFrame.markerPoints
		self.numSegments = self.imFrame.currentMarker

		#Empty list
		if(self.result == [[]]):
			self.result = 0

		self.imFrame.grid_forget()
		self.bottomFrame.grid_forget()
		self.imFrame.destroy()
		self.bottomFrame.destroy()

		self.grid_forget()
		self.destroy()

		self.root.destroy()


class ImageFrame(tk.Frame):

	"""
	Frame containing an interactive image for marker point selection
	"""

	def __init__(self, parent, imgPath, maxWidth=1400, maxHeight=650):
		self.parent = parent
		tk.Frame.__init__(self, parent)

		self.imgCV = cv2.imread(imgPath)
		self.img_original = self.imgCV.copy()
		self.img = ImageTk.PhotoImage(Image.open(imgPath))
		self.markerPoints = [[]]
		self.currentMarker = 1

		self.canvas = tk.Canvas(self)
		
		if(maxHeight > 0 and self.img.height() > maxHeight):
			self.canvas['height'] = maxHeight
		else:
			self.canvas['height'] = self.img.height()

		if(maxWidth > 0 and self.img.width() > maxWidth):
			self.canvas['width'] = maxWidth
		else:
			self.canvas['width'] = self.img.width()
		
		self.canvasImage = self.canvas.create_image(0,0,image=self.img, anchor="nw")
		self.canvas.grid(row=0, column=0)
		self.canvas['scrollregion']= (0,0,self.img.width(),self.img.height())

		self.canvas.bind("<Button-1>", self.leftclick)
		self.canvas.bind("<B1-Motion>", self.leftclick)

		yScrollbar = tk.Scrollbar(self,orient="vertical",command=self.canvas.yview)
		yScrollbar.grid(row=0, column=1, sticky="ns")

		xScrollbar = tk.Scrollbar(self,orient="horizontal",command=self.canvas.xview)
		xScrollbar.grid(row=1, column=0, sticky="we")

		self.canvas['yscrollcommand'] = yScrollbar.set
		self.canvas['xscrollcommand'] = xScrollbar.set


	def cleanup(self):

		"""
		Reset markerPoints and markerPoint iterator
		"""

		self.markerPoints = [[]]
		self.currentMarker = 1

	def leftclick(self,event):

		"""
		Handle a left click on the image made by the user.
		Mark points for easy visualisation, and also save the coordinates of the marked points
		in the markerPoints list
		"""

		x,y = int(self.canvas.canvasx(event.x)-1), int(self.canvas.canvasy(event.y)-1)

		if(x > 0 and x < self.imgCV.shape[1] and y>0 and y < self.imgCV.shape[0]):
			self.markerPoints[self.currentMarker-1].append((y, x))
			self.addMarker(x, y)
			self.updatePhotoImgFromCVImg()

	def addMarker(self, x, y):

		"""
		Adds a filled red circle and currentMarker number at the given x,y coordinate

		Args:
			x (int): The x-coordinate in the image matrix (Not imDisplay frame)
			y (int): The y-coordinate in the image matrix (Not imDisplay frame)
		"""

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(self.imgCV, "{}".format(self.currentMarker),(x,y), font, 0.5,(255,255,255),1,cv2.LINE_AA)
		
		cv2.circle(self.imgCV,(x,y), 1, (0,0,255), -1)
		

	def updatePhotoImgFromCVImg(self):

		"""
		Updates the self.img using the self.imgCV
		"""

		#Reasamble the image from BGR to RGB format
		b,g,r = cv2.split(self.imgCV)
		imgRGB = cv2.merge((r,g,b))

		#Convert to PhotoImage
		imgRGB = Image.fromarray(imgRGB)
		self.img = ImageTk.PhotoImage(image = imgRGB)
		self.canvas.itemconfig(self.canvasImage, image = self.img)


class BottomFrame(tk.Frame):

	"""
	Interactive frame below the imageframe for easy navigation of next object, reset
	objects, and exit gui/display result
	"""

	def __init__(self, parent):
		self.parent = parent
		tk.Frame.__init__(self, parent)

		#Buttons
		tk.Button(self, command=lambda:self.nextObject(), text="Next object").grid(row=2, column=0)
		tk.Button(self, command=lambda:self.resetObjects(), text="Reset objects").grid(row=3, column=0)
		tk.Button(self, command=lambda:self.parent.exitGUI(), text="Display result").grid(row=3, column=1)
		tk.Button(self, command=lambda:self.resetLastObject(), text="Reset last object selection").grid(row=2, column=1)

		#Label
		self.botLabel = tk.Label(self, text="\n Number of segments selected: {}\n".format(self.parent.imFrame.currentMarker))
		self.botLabel.grid(row=0, columnspan=2)


	def nextObject(self):

		"""
		Select next object
		"""

		self.parent.imFrame.markerPoints.append([])
		self.parent.imFrame.currentMarker += 1
		self.botLabel['text'] = "\n Number of segments selected: '{}'\n".format(self.parent.imFrame.currentMarker)

	def resetObjects(self):

		"""
		Restart object selection
		"""

		self.parent.imFrame.markerPoints = [[]]
		self.parent.imFrame.currentMarker = 1
		self.parent.imFrame.imgCV = self.parent.imFrame.img_original.copy()
		self.parent.imFrame.updatePhotoImgFromCVImg()
		#self.parent.imFrame.imDisplay['image'] = self.parent.imFrame.img
		self.botLabel['text'] = "\n Number of segments selected: '{}'\n".format(self.parent.imFrame.currentMarker)

	def resetLastObject(self):

		"""
		Remove last object selection/creation in GUI
		"""
		
		if(self.parent.imFrame.currentMarker == 1):
			self.resetObjects()
			return
			
		del self.parent.imFrame.markerPoints[-1]

		self.parent.imFrame.imgCV = self.parent.imFrame.img_original.copy()
		
		markerPoints = self.parent.imFrame.markerPoints
		for i in range(len(markerPoints)):
			self.parent.imFrame.currentMarker = i+1
			for j in range(len(markerPoints[i])):
				self.parent.imFrame.addMarker(markerPoints[i][j][1], markerPoints[i][j][0])
		
		self.parent.imFrame.updatePhotoImgFromCVImg()
		self.botLabel['text'] = "\n Number of segments selected: '{}'\n".format(self.parent.imFrame.currentMarker)









