from ij import IJ
imp = IJ.getImage()
print imp

from ij.io import FileSaver
fs = FileSaver(imp)
fs.save()

folder = "Z:\\CodeCluster\\SampleMicroPics"
filepath = folder + (immortalized_human_astrocytes.jpeg)
fs.saveAsPng(filepath)

folder = "Z:\\CodeCluster\\ProcessedPics"

from os import path

if path.exists(folder) and path.isdir(folder):
	print "folder exists:", folder
	filepath = path.join(folder, (immortalized_human_astrocytes.jpeg))
	if path.exists(filepath):
		print (File exists! Not saving the image, would overwrite a file!)
	elif fs.saveAsTiff(filepath):
		print (File saved successfully at), filepath	
else:
	print (Folder does not exist or it's not a folder!)