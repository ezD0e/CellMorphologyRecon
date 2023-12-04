from ij import IJ
import os

folder = 'Z:\\CodeCluster\\SampleMicroPics'
for i, filename in enumerate(os.listdir(folder)):
    img = IJ.openImage(os.path.join(folder, filename))
#    if img is None
#       print "Could not open image from file:", filename
#       continue
#   img-IJ.getImage()
#   print(img)
#   exit
    IJ.run(img,"8-bit","");
#   IJ.run(img,"Sharpen","");
#   IJ.run(img,"Gaussian Blur...","sigma=1");
    IJ.run(img,"Enhance Contrast","saturated=0.35");
    IJ.setAutoThreshold(img,"Default");
    IJ.run(img,"Convert to Mask","");
    IJ.run(img,"Fill Holes","");
    IJ.run(img,"Analyze Particles...", "size=30-Infinity show=Outlines exclude add");

#   IJ.save(img,'Z:\\CodeCluster\\ProcessedPics\\img_%s.tif'%i)

    IJ.roiManager(img,"Measure")
    IJ.saveAs("Results", 'Z:\\CodeCluster\\ProcessedPics\\img_%s.csv'%i');