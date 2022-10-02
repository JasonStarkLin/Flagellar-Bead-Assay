import fnmatch
import os
import shutil
import numpy as np
import pims
import matplotlib.pyplot as plt

from skimage.segmentation import clear_border
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import pandas as pd
import Fun_BeadAssay as BA
import time



from matplotlib.patches import Ellipse

def DrawTrace(Data,Speed_Sheet,FT,filepath,title):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(Data[0], Data[1], 'ro', label='test data', zorder=1, markersize=1)
    e_cen = [Speed_Sheet.loc[0,"e-x"],Speed_Sheet.loc[0,"e-y"]]
    width = Speed_Sheet.loc[0,"e-major"]
    height = Speed_Sheet.loc[0,"e-minor"]
    phi = Speed_Sheet.loc[0,"e-angle"]
    ellipse = Ellipse(xy=e_cen, width=2 * width, height=2 * height, angle=np.rad2deg(phi),
                      edgecolor='b', fc='None', lw=2, label='Fit', zorder=2)
    ax[0].add_patch(ellipse)
    ax[0].set_title(title)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xlim((0,12))
    ax[0].set_ylim((0,12))


    ax[1].scatter(FT[0],FT[1],s=1,color='k')
    ax[1].set_xlabel("Speed(Hz)")
    ax[1].set_title("FFT results")
    #plt.tight_layout()
    max_ind = np.where(FT[1]==np.max(FT[1]))
    max_amp = FT[1][max_ind[0]][0]
    max_F = FT[0][max_ind[0]][0]
    s = "["+"{:.0f}".format(max_F)+","+"{:.1f}".format(max_amp)+"]"
    ax[1].text(max_F,max_amp,s,color='c',horizontalalignment="left",verticalalignment="top")

    #plt.draw()
    fig.savefig(filepath)
    plt.close()

def ExFolderConstr(Con_folder):
    if os.path.exists(Con_folder):
        print("Analysis Exporting folder have been existed")
        print("Deleted the Analysis exporting folder and Created a new one")
        shutil.rmtree(Con_folder)
        os.mkdir(Con_folder)

    else:
        print("Analysis Exporting folder is not exist")
        os.mkdir(Con_folder)
        print("Created the analysis folder")



folder = "C:\\Users\\LTS\\Desktop\\Bead Assay Data\\"
ExpFolder = "C:\\Users\\LTS\\Desktop\\Bead Assay Data\Analysis\\"
SampleName = '20201022-BE-Ibidi-853085-S23'

#folder = "D:\\NAS-TEMP_BE\\SpeedDistribution Debug\\"
#SampleName = 'test'
#ExpFolder = "D:\\NAS-TEMP_BE\\SpeedDistribution Debug\\test\\"

FPS = 451
N_frame = 450 # for one analysis section.
window_size = 6 #my setting is 6

temp_time = time.time()
#Searching files.
for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*.seq'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
print(Targetfile)




#Check and Creating the exporting file folder
FigFolder = ExpFolder+SampleName+"-OrbitResults\\"
B_Label_Folder = ExpFolder+SampleName+"-Bead_Label\\"
ExFolderConstr(FigFolder)
ExFolderConstr(B_Label_Folder)

Bead_sheet = pd.DataFrame(columns=["Label","DateTime","X","Y","Speed(Hz)","FFT_peak","FFT_amp","FFT_pr","e-x","e-y","e-major",
                                   "e-minor","e-angle","e-FitQaulity","Eccentricity","AspectRatio","Opening angle(dgree)","Radius"])
for num,i in enumerate(Targetfile):

    #Open and input the image file
    Image = pims.open(folder+i)
    ImShape = np.shape(Image[0])
    Split_FileName = i.rsplit('.')
    Sample = Split_FileName[0]
    print("File: ",num, i)
    print("     Frame Shape", np.shape(Image))
    print("     Image type: ", Image.pixel_type)
    print("     Frame Counts: ", len(Image))

    #Segment the likely rotating beads
    ImageSTD = np.std(Image[0:100], axis=0)
    ThreImage = ImageSTD > np.mean(ImageSTD) + 7*np.std(ImageSTD) #The value 7 in algorithm is derived from experience.
    CloImage = closing(ThreImage,square(6))
    BI = clear_border(CloImage)
    Label_image = sk_label(BI)
    Image_label_overlay = label2rgb(Label_image, image=ThreImage,bg_label=0)


    fig,ax = plt.subplots(figsize=(20.48,3.5))
    ax.imshow(Image[num],cmap="gray")
    for label,region in enumerate(regionprops(Label_image)):
        center = region.centroid
        ax.scatter(center[1],center[0],facecolors='none',edgecolor='g')
        Bead_Name = str("{:0>2d}".format(label))
        ax.text(center[1], center[0], Bead_Name,fontsize=8, ha="right", va="bottom", color='yellow')
    fig.savefig(B_Label_Folder+Sample + ".png")
    plt.close(fig)
    #plt.show()

    #continue

    for label,region in enumerate(regionprops(Label_image)):
        center = region.centroid
        #ax.scatter(center[1],center[0],facecolors='none',edgecolor='g')
        ROIName = '-Bead-'+str("{:0>2d}".format(label))
        KeyLabel = Sample+ROIName
        bead_xy = np.array([int(center[1]), int(center[0])])
        if BA.EdgeTest(bead_xy,ImShape,Extensize= window_size):
            print("     ", KeyLabel)
            Bead_P = BA.GetBeadsPosition(folder, [i], window_size, bead_xy, ImageSave=False, Bead_label=ROIName)
            Position_sheet = pd.DataFrame(Bead_P,columns=['Frames','DateTime', 'x-Center', 'y-Center'])
            Position_sheet = Position_sheet.infer_objects()
            xy_data = [Position_sheet["x-Center"].to_numpy(), Position_sheet["y-Center"].to_numpy()]
            Speed_Sheet,FT_Data = BA.GetSpeed(Position_sheet, Steps_N=N_frame, Fps=FPS,FT_Output=True)
            #Plot orbit information
            title = "{:.1f}".format(Speed_Sheet.loc[0,"Speed(Hz)"]) + "Hz"\
                    +", AR:" + "{:.1f}".format(Speed_Sheet.loc[0,"AspectRatio"]) \
                    +", Eccen:"+"{:.1f}".format(Speed_Sheet.loc[0,"Eccentricity"])\
                    +"\n, Radius:"+"{:.2f}".format(Speed_Sheet.loc[0,"Radius"]) \
                    +", Ell_Q:"+"{:.1f}".format(Speed_Sheet.loc[0,"e-FitQaulity"])
            filepath = FigFolder + KeyLabel + ".jpg"
            #DrawTrace function need to modify.
            DrawTrace(xy_data,Speed_Sheet,FT_Data,filepath,title)  # If results contain not only one speed. The results would be wrong. Need to upgrade.
            #Try to improve the code for temp_Sheet if possible.
            temp_Sheet = pd.Series({"Label": KeyLabel,
                                    "DateTime":Speed_Sheet.loc[0,"DateTime"],
                                    "X": center[1], "Y": center[0],
                                    "Speed(Hz)": Speed_Sheet.loc[0,"Speed(Hz)"],
                                    "FFT_peak": Speed_Sheet.loc[0,"FFT_peak"],
                                    "FFT_amp": Speed_Sheet.loc[0,"FFT_amp"],
                                    "FFT_pr": Speed_Sheet.loc[0,"FFT_pr"],
                                    "e-x":Speed_Sheet.loc[0,"e-x"],
                                    "e-y":Speed_Sheet.loc[0,"e-y"],
                                    "e-major":Speed_Sheet.loc[0,"e-major"],
                                    "e-minor":Speed_Sheet.loc[0,"e-minor"],
                                    "e-angle":Speed_Sheet.loc[0,"e-angle"],
                                    "e-FitQaulity":Speed_Sheet.loc[0,"e-FitQaulity"],
                                    "Eccentricity":Speed_Sheet.loc[0,"Eccentricity"],
                                    "AspectRatio":Speed_Sheet.loc[0,"AspectRatio"],
                                    "Opening angle(dgree)": Speed_Sheet.loc[0,"Opening angle(dgree)"],
                                    "Radius":Speed_Sheet.loc[0,"Radius"]
                                    })
            Bead_sheet = Bead_sheet.append(temp_Sheet, ignore_index=True)


Bead_sheet = Bead_sheet.set_index("Label")
ExpBeadFileName = ExpFolder + SampleName +'.csv'
Bead_sheet.to_csv(ExpBeadFileName)
print("Cost Time: ",(time.time()-temp_time)/60," Minutes")