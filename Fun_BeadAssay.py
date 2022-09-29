"Ver 1.0"
import numpy as np
import pims
from skimage.io import imsave
import sys
import pandas as pd
from scipy.fftpack import fft
import ellipses as el
import traceback
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.constants as SciCon

"""Functions for beads assay data analysis
    __author__ = "Tsai-Shun Lin"
    __credits__ = [""]
    __maintainer__ = "Tsai-Shun"
    __email__ = "tsaishun.lin@gmail.com"
    __status__ = "Development"
    Requirements 
    ------------
    Python 3.X
    numpy
    scikit-image
    pandas
    matplotlib
    ellipses
    
    References
    ----------
"""


def progressBar(value, endvalue,ProText = "Progress:", bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r      "+ProText+" [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def GetBeadsPosition(folder,Targetfile, window_size, bead_xy, **kwargs):
    '''

    Parameters
    ----------
    folder : string
        The destination for the files position.
    Targetfile : list
        The list includes all the name of the files which intend to calculate the position. The filename older in the list is the iterate sequence.
    window_size : int or float
        The number that indicate how large the extended size to crop the images to calculate the position.
    bead_xy : 1d-array.
        A numpy array included 2D central posiiton of the target. (x,y)
    kwargs

    Returns
    -------
    2d array
        The beads informations. Including:
        frame_count : the position of the slice.
        Time : The time data for that position.
        x: x position in the Image.
        y: y position in the Image.

    '''
    N_Frames = 0
    for i in Targetfile:
        Temp_image = pims.open(folder + i)
        N_Frames += len(Temp_image)
    Im_type = Temp_image.pixel_type
    Crop_images = np.zeros((N_Frames, 2 * window_size + 1, 2 * window_size + 1), dtype=Im_type)
    frame_count = 0
    # Run for all the files
    # Import the transfer function for the beads position calibration.
    if "TranFun" in kwargs:
        TranFun = kwargs["TranFun"]
    for i in range(0,len(Targetfile)):
        Image = pims.open(folder + Targetfile[i])
        Image_Frames = len(Image)

        #print("     File: ", Targetfile[i])
        #print("     Frame Shape", np.shape(Image))
        #print("     Image type: ", Image.pixel_type)
        #print("     Frame Counts: ", Image_Frames)
        # Run for the frames
        #for j in range(0, 450):
        for j in range(Image_Frames):
            #if frame_count<41400 :
                #frame_count = frame_count + 1
                #continue
            #if j != 3339 : continue
            progressBar(100 * frame_count / N_Frames, 100)

            if "TranFun" in kwargs:
                #Timepoint = (frame_count//450)-1  # Because the imagej did not record transfer function for the first frame.
                Timepoint = frame_count  # Correct for each frame. The Tranfer function is not the same as Imagej. Need to be careful
                if Timepoint >=0:
                    Crop_cent_x = np.int(bead_xy[0] - TranFun.iloc[Timepoint,2])
                    Crop_cent_y = np.int(bead_xy[1] - TranFun.iloc[Timepoint,3])
                else:
                    Crop_cent_x = bead_xy[0]
                    Crop_cent_y = bead_xy[1]
            else:
                Crop_cent_x = bead_xy[0]
                Crop_cent_y = bead_xy[1]

            Crop_images[frame_count, :, :] = Image[j][Crop_cent_y - window_size:Crop_cent_y + window_size + 1,
                                       Crop_cent_x - window_size:Crop_cent_x + window_size + 1]
            #Kick out background
            temp = Crop_images[frame_count, :, :] * (Crop_images[frame_count, :, :] > np.average(Crop_images[frame_count, :, :]) + np.std(Crop_images[frame_count, :, :]))
            Cent = SpotCentral(temp, fun='Centroid')

            #this part may be able to boost up. The np.append may slow down the calculation.
            if "time" in Image[j].metadata :
                DTime = Image[j].metadata['time']
            else :
                DTime = frame_count

            if 'Bead_P' in locals():
                Bead_P = np.append(Bead_P, np.array([[frame_count,DTime, Cent[0], Cent[1]]]), axis=0)
            else:
                Bead_P = np.array([[frame_count,DTime, Cent[0], Cent[1]]])

            # For saving the images.
            Cx = int(np.round(Cent[0]))
            Cy = int(np.round(Cent[1]))
            Crop_images[frame_count, Cy, Cx] = 255
            frame_count = frame_count + 1
        print("")
    #print('     Totally frames number : ', frame_count)
    print("")
    #save the images
    if kwargs["ImageSave"] is True:
        Split_FileName = Targetfile[0].rsplit('-', 2)
        Sample = Split_FileName[0]
        imsave(ExpFolder + Sample + '-Fitted' + kwargs['Bead_label'] + '.tif', Crop_images[0:frame_count, :, :], photometric='minisblack')
    return Bead_P

def GetSpeed(PositionData, Steps_N, Fps,FT_Output=False,**kwargs):
    #PositionData.reset_index(drop=True)
    P_cmp = 1j * PositionData["y-Center"]
    P_cmp = P_cmp + PositionData["x-Center"]  # position complex Z = x +yi
    N_Speed = len(P_cmp) // Steps_N
    SpeedSheet = pd.DataFrame(columns=["Frame", "DateTime", "Speed(Hz)"])
    #print(len(P_cmp))

    for i in range(N_Speed):
        Start_ind = i*Steps_N
        End_ind = (i+1)*Steps_N
        Temp_Pdata = P_cmp[Start_ind:End_ind]
        ave_FFT_P = np.average(Temp_Pdata)
        Normal_factor = 2/len(Temp_Pdata)
        FFT_P = abs(fft(Temp_Pdata-ave_FFT_P)*Normal_factor) #Calculate FFT and normalize the amplitude.
        fft_ps = abs(fft(Temp_Pdata-ave_FFT_P))**2 #Power spectrum.
        FFT_pw = fft_ps/np.sum(fft_ps) #Frequncy power spectrum normalize by the total power. (frequency weighting)
        freq = np.fft.fftfreq(Temp_Pdata.size,1/Fps) #Generate the frequence sequence for FFT results.
        freq_max_ind = np.where(FFT_P == np.max(FFT_P)) #Find the position of the maximum in FFT results.
        #print('Speed (Hz) :',freq[freq_max_ind[0]])
        FFT_Amp = FFT_P[freq_max_ind[0]][0]
        Speed = freq[freq_max_ind[0]][0]
        FFT_pw_max = FFT_pw[freq_max_ind[0]][0]
        Frame = PositionData.loc[Start_ind, "Frames"]
        DateTime = PositionData.loc[Start_ind, "DateTime"]


        #Tstage = PositionData.iloc[Start_ind,2]


        ####### This part may still need to modify########################################################
        try:
            # Fit ellipse
            lsqe = el.LSqEllipse()
            Data = [PositionData.loc[Start_ind:End_ind,"x-Center"].to_numpy(), PositionData.loc[Start_ind:End_ind,"y-Center"].to_numpy()]
            lsqe.fit(Data)
            e_cen, width, height, phi = lsqe.parameters()
            if np.complex128 in [type(j) for j in [e_cen, width, height, phi]]:
                # The ellipse fitting using numpy function "numpy.linalg.eig" which may return a complex number.
                # for programing furthur process, I need make the complex number back to real number.
                e_cen = np.real(e_cen)
                width, height, phi = np.real([width, height, phi])

            # Calculate the eccentricity.
            Eccen = GetEccentricity(width, height)
            Ellipse_Q = GetEllipseQ(Data, e_cen, width, height, phi)  # Calculate the fitting quality.
            Ar = GetAspectRatio(width, height)  # Aspect ratio
            OpenAngle = np.arccos(Ar) * 180 / np.pi  # Calculating the angle tilt from the z axis
            radius = np.max([width, height])

            s = pd.Series({"Frame":Frame,
                           "DateTime": DateTime,
                           "Speed(Hz)": abs(Speed),
                           "FFT_peak": Speed,
                           "FFT_amp": FFT_Amp,
                           "FFT_pr":FFT_pw_max, #FFT power spectrum ratio
                           "e-x": e_cen[0],
                           "e-y": e_cen[1],
                           "e-major": width,
                           "e-minor": height,
                           "e-angle": phi,
                           "e-FitQaulity": Ellipse_Q,
                           "Eccentricity": Eccen,
                           "AspectRatio": Ar,
                           "Opening angle(dgree)": OpenAngle,
                           "Radius": radius
                                    })
        except:
            # if the ellipse fit failed -> fill np.NAN to the related data.
            traceback.print_exc()
            print("Elipse fit failed at i :",i)
            print("Start index : Final index: ",Start_ind,End_ind)
            s = pd.Series({"Frame":Frame,
                           "DateTime": DateTime,
                           "Speed(Hz)": abs(Speed),
                           "FFT_peak": Speed,
                           "FFT_amp": FFT_Amp,
                           "FFT_pr": FFT_pw_max,
                           "e-x": np.NaN,
                           "e-y": np.NaN,
                           "e-major": np.NaN,
                           "e-minor": np.NaN,
                           "e-angle": np.NaN,
                           "e-FitQaulity": np.NaN,
                           "Eccentricity": np.NaN,
                           "AspectRatio": np.NaN,
                           "Opening angle(dgree)": np.NaN,
                           "Radius": np.NaN
                                    })
        ################################################################################################################

        #s = pd.Series({"Frame": Frame, "DateTime": DateTime, "Speed(Hz)": abs(Speed), "FT_amp": FFT_Amp})
        SpeedSheet = SpeedSheet.append(s, ignore_index=True)

    if FT_Output == True:
        return SpeedSheet,[freq,FFT_P]
    else:
        return SpeedSheet

def EllipseCorrection(PositionData):
    ##Not finished yet. Have bug in OpenAngle correction. -> I decided to simply the method. Direct calculate the ratio (C_ratio). And it works.
    ##Still need to clean the code
    ##If the ellipse fitting failed, It would skip the correction.


    # Fit ellipse
    lsqe = el.LSqEllipse()
    Data = [PositionData.loc[:, "x-Center"].to_numpy(),
            PositionData.loc[:, "y-Center"].to_numpy()]
    try:
        lsqe.fit(Data)
    except:
        print("Ellipse fit error")
        return PositionData
    e_cen, width, height, phi = lsqe.parameters()
    # Calculate the eccentricity.
    Eccen = GetEccentricity(width, height)
    Ellipse_Q = GetEllipseQ(Data, e_cen, width, height, phi)  # Calculate the fitting quality.
    Ar = GetAspectRatio(width, height)  # Aspect ratio
    OpenAngle = np.rad2deg(np.arccos(Ar))   # Calculating the angle tilt from the z axis
    radius = np.max([width, height])



    '''**********Circular Correction*******************************'''
    # Align the long axis angle
    if width > height:
        C_ratio = width/height
        AlignAngle = 90 + np.rad2deg(phi)
        #print("width > height", AlignAngle)
    else:
        C_ratio = height / width
        AlignAngle = np.rad2deg(phi)
        #print("width <= height", AlignAngle)

    P = PositionData.loc[:, ["y-Center","x-Center"]].to_numpy()
    P = np.append(np.zeros((len(PositionData),1)),P, axis=1)

    r = Rot.from_euler('zyx', [0, 0, AlignAngle], degrees=True)
    Alin_p = r.apply(P)
    PositionData.loc[:, "x-Center"] = Alin_p[:, 2]*C_ratio
    PositionData.loc[:, "y-Center"] = Alin_p[:, 1]
    return PositionData
    #r = Rot.from_euler('zyx', [0, -OpenAngle, 0], degrees=True)
    #print("Open Angle = ",OpenAngle)
    #Cor_p = r.apply(Alin_p)
    #ax3.scatter(Cor_p[:, 2], Cor_p[:, 1], color="b", s=4)
    #'''**************************************************************'''


    '''**********Plot the circular correction results.****************'''
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_aspect("equal")
    ax1.scatter(PositionData.loc[:, "x-Center"].to_numpy(), PositionData.loc[:, "y-Center"].to_numpy(), color="y", s=4, alpha=0.2)
    ellipse = Ellipse(xy=e_cen, width=2 * width, height=2 * height, angle=np.rad2deg(phi),
                      edgecolor='b', fc='None', lw=2, label='Fit', alpha=0.5)
    ax1.add_patch(ellipse)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_aspect("equal")
    ax2.scatter(Alin_p[:, 2], Alin_p[:, 1], color="b", s=4)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_aspect("equal")
    ax3.scatter(PositionData.loc[:, "x-Center"], PositionData.loc[:, "y-Center"], color="b", s=4)
    plt.show()
    #'''**************************************************************'''

def GetSpeed_Ins(PositionData, Steps_N, Fps):
    PositionData.reset_index(drop=True)

    #Scale X,Y signal from -1 to +1
    Px = PScale(PositionData.loc[:, "x-Center"]).to_numpy()
    #Px = np.reshape(Px, (len(Px), 1))
    Py = PScale(PositionData.loc[:, "y-Center"]).to_numpy()
    #Py = np.reshape(Py,(len(Py),1))

    #Calculate the radius to the center
    x2 = (PositionData.loc[:, "x-Center"]-np.mean(PositionData.loc[:, "x-Center"]))**2
    y2 = (PositionData.loc[:, "y-Center"]-np.mean(PositionData.loc[:, "y-Center"]))**2
    Cen_distance = np.sqrt(x2+y2)

    P_r = np.sqrt(Px**2 + Py**2)
    P_theta = np.arctan2(Py,Px)
    P_theta[P_theta<0] = P_theta[P_theta<0]+2*np.pi  #Correct the angle from 0 to 2 pi
    P_CumAngle = AngleCumlative(P_theta)  #While in this step the P_theta was rewrite as well.

    deltaAngle = P_CumAngle[1::]-P_CumAngle[0:-1]
    deltaT = 1/Fps
    Rotation_speed_w = deltaAngle/deltaT
    Rotation_speed_Hz = Rotation_speed_w/(2*np.pi)

    SpeedLength = len(PositionData)-1
    SpeedSheet = pd.DataFrame({"Frame":PositionData.loc[0:SpeedLength-1,"Frames"], # in dataframe the last index also count
                               "DateTime":PositionData.loc[0:SpeedLength-1,"DateTime"],
                               "Polar_r":P_r[0:SpeedLength], # In Np array the last index does not count.
                               "Polar_Theta":P_CumAngle[0:SpeedLength],
                               "Centroid distance":Cen_distance[0:SpeedLength],
                               "Speed(Hz)":Rotation_speed_Hz,
                               "Angular Speed(rad/s)":Rotation_speed_w
                               })
    return SpeedSheet
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_aspect("equal")
    ax1.scatter(Px,Py, color="y", s=4, alpha=0.2)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(SpeedSheet["Frame"],SpeedSheet["Speed(Hz)"], color="y", s=4, alpha=0.2)
    plt.show()
    '''

def AngleCumlative(Thetas):
    for i in range(len(Thetas)-1):
        if Thetas[i+1]-Thetas[i] > np.pi/2:
            Thetas[i+1::] = Thetas[i+1::] - 2*np.pi
        elif Thetas[i+1]-Thetas[i] < -1*np.pi/2:
            Thetas[i+1::] = Thetas[i + 1::] + 2*np.pi
    return Thetas

def SmoothSpeed(SpeedSheet,Smooth_Np):
    LoopSize = len(SpeedSheet) // Smooth_Np
    Ave_SpeedSheet = pd.DataFrame(columns=SpeedSheet.columns)
    for i in range(LoopSize):
        Start_ind = i*Smooth_Np
        End_ind = Start_ind + Smooth_Np-1
        M_Speed = np.mean(SpeedSheet.loc[Start_ind:End_ind,"Speed(Hz)"])
        s = SpeedSheet.loc[Start_ind, :].copy()
        s.loc["Speed(Hz)"] = M_Speed
        Ave_SpeedSheet = Ave_SpeedSheet.append(s, ignore_index=True)
    return Ave_SpeedSheet

def PScale(Positions):  # Normalized the Positions between -1  to +1.
    Max = np.max(Positions)
    Min = np.min(Positions)
    S_P = (Positions-Min)*2/(Max-Min) - 1
    return (S_P)

def SpotCentral(Image,**kwargs):
    '''THis function will fit the image as my define'''
    try:
        #Centroid method
        if kwargs['fun'] == 'Centroid':
            Cent = CentOfMass(Image)
        elif kwargs['fun'] == 'npGaussian':
            #Use numerical python Gaussian fitting
            #Learn from: https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
            try:
                max_p = np.where(Image == np.max(Image))
                initial_guess = (np.max(Image), max_p[1][0], max_p[0][0], 5, 5, 0, np.min(Image))
                height, width = np.shape(Image)
                fit_bound = ((0, 0, 0, 0, 0, 0, 0),
                             (np.inf, width, height, width/2.0, height/2.0, 2 * np.pi, 255))
                x = np.linspace(0, width-1,width)
                y = np.linspace(0, height-1,height)
                x, y = np.meshgrid(x, y)
                popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), Image.flatten(), p0=initial_guess, bounds=fit_bound)
                Cent = (popt[1], popt[2])
            except:
                Cent = (max_p[1][0], max_p[0][0])
        elif kwargs['fun'] == 'astropyGaussian':
            #Use the 2D gaussain fit learn from astropy.
            Gp, Gp_image = Gaussian2Dfit(Image)
            Cent = (Gp._parameters[1],Gp._parameters[2])
        else:
            print("No such Method in my function")
            print("Try following: Centroid, npGaussian, astropyGaussian")
    except:
        Cent = CentOfMass(Image)

    return Cent

def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)+ c*((y-yo)**2)))
    return g.ravel()

def CentOfMass(Image):
    size_y, size_x = np.shape(Image)
    x = np.linspace(1,size_x,size_x) # from 1 in order to avoid position 0.
    y = np.linspace(1,size_y,size_y)
    x, y = np.meshgrid(x, y)
    Int_sum = np.sum(Image)
    if Int_sum == 0:
        print("SUM of image intensity is 0. Error")
        return(0,0)
    Cen_x = (np.sum(x*Image)/Int_sum) -1 # minus one in order to shift back position which cause by in_weight_x
    Cen_y = (np.sum(y*Image)/Int_sum) -1

    return (Cen_x, Cen_y)

def Gaussian2Dfit(Image):
    size_y, size_x = np.shape(Image)
    y, x = np.mgrid[:size_y, :size_x]
    max_p = np.where(Image == np.max(Image))
    #initial_guess = (np.max(Image), max_p[1][0], max_p[0][0], 5, 5, 0, np.min(Crop_image))

    p_init = models.Gaussian2D(amplitude=np.max(Image), x_mean=max_p[1][0], y_mean=max_p[0][0])
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, Image)
    return p, p(x,y)

def EdgeTest(Spot_xy, Shape, Extensize = 0):
    Left = Spot_xy[0] - Extensize > 1
    Top = Spot_xy[1] - Extensize > 1
    Right = Spot_xy[0] + Extensize < Shape[1] - 1
    Bottom = Spot_xy[1] + Extensize < Shape[0] - 1
    return Left*Top*Right*Bottom

def GetEccentricity(width, height):
    if width > height:
        a=width
        b=height
    else:
        b=width
        a=height

    c= np.sqrt(a**2-b**2)
    return (c/a)

def GetEllipseQ(Data,e_cen, width, height, phi):
    Xd = Data[0]-e_cen[0]
    Yd = Data[1]-e_cen[1]
    C1 = ((Xd * np.cos(phi) + Yd * np.sin(phi))**2)/(width**2)
    C2 = ((Xd * np.sin(phi) - Yd * np.cos(phi))**2)/(height**2)
    Q = np.sum((C1+C2-1)**2)
    return Q

def GetAspectRatio(width,height):
    if width > height:
        return(height/width)
    else:
        return(width/height)

def FFT_Results(PositionData, Fps):
    #This function use all the positiondata to calculate FFT. And return the FFT profile.
    # PositionData.reset_index(drop=True)
    P_cmp = 1j * PositionData["y-Center"]
    P_cmp = P_cmp + PositionData["x-Center"]  # position complex Z = x +yi


    Temp_Pdata = P_cmp
    ave_FFT_P = np.average(Temp_Pdata)
    Normal_factor = 2 / len(Temp_Pdata)
    FFT_P = abs(fft(Temp_Pdata - ave_FFT_P) * Normal_factor)  # Calculate FFT and normalize the amplitude.
    freq = np.fft.fftfreq(Temp_Pdata.size, 1 / Fps)  # Generate the frequence sequence for FFT results.
    #freq_max_ind = np.where(FFT_P == np.max(FFT_P))  # Find the position of the maximum in FFT results.
    # print('Speed (Hz) :',freq[freq_max_ind[0]])
    #FFT_Amp = FFT_P[freq_max_ind[0]][0]
    #Speed = freq[freq_max_ind[0]][0]

    return freq, FFT_P

def GetOrbInfo(Xdata,Ydata):
    Data = [Xdata,Ydata]
    try:
        # Fit ellipse
        lsqe = el.LSqEllipse()
        lsqe.fit(Data)
        e_cen, width, height, phi = lsqe.parameters()

        # Calculate the eccentricity.
        Eccen = GetEccentricity(width, height)
        Ellipse_Q = GetEllipseQ(Data, e_cen, width, height, phi)  # Calculate the fitting quality.
        Ar = GetAspectRatio(width, height)  # Aspect ratio
        OpenAngle = np.arccos(Ar) * 180 / np.pi  # Calculating the angle tilt from the z axis
        radius = np.max([width, height])
        results = {"e-x": e_cen[0], "e-y": e_cen[1], "e-major": width, "e-minor": height, "e-angle": phi,
                  "e-FitQaulity": Ellipse_Q,
                  "Eccentricity": Eccen,
                  "AspectRatio": Ar,
                  "Opening angle(dgree)": OpenAngle,
                  "Radius": radius
                  }
    except:
        # if the ellipse fit failed -> fill np.NAN to the related data.
        traceback.print_exc()
        print("Elipse fit failed")
        results = {"e-x": np.NaN, "e-y": np.NaN, "e-major": np.NaN, "e-minor": np.NaN, "e-angle": np.NaN,
                  "e-FitQaulity": np.NaN,
                  "Eccentricity": np.NaN,
                  "AspectRatio": np.NaN,
                  "Opening angle(dgree)": np.NaN,
                  "Radius": np.NaN
                  }
    return results

def GetTorque(f, re, r_bead=495, visco=9.86e-10, Drag_fila=0.8, Faxen=False, d_wall=380):
    '''
    This function is used to calculate the torque from the input speed and beads information.
    Also, I integrate the Faxen correction to the function. But, from my test, the correction may have problem.
    As d_wall too samll ~ 5 nm as in Ashley paper, the drag coefficient is too samll (near zero.).
    I think I need to doulble check it.

    Parameters
    ----------
    f: Float number or np.array.
        Rotational Speed.
        unit: Hz
    r_bead: Float number.
        Radius of the bead.
        unit: nm
    re: Float number.
        Rotational enccentricity of the bead. It is measured radial distance to the bead’s axis of rotation (Nord,2017,PNAS).
        unit: nm
    visco: Float number.
        Viscosity of the medium.
        Unit: pN s/ nm^2.
        The default value is 9.86e-10 which taken from Inoue,2008,JMB.
    Drag_fila: Float number.
        Drag coefficient from the stub of the filament.
        Unit: pN nm s/rad.
        The default value is 0.8 which taken from Inoue,2008,JMB.
        The value get under condition: shear 60 times. 26-gauge needles connected by a piece of polyethylene tubing (12 cm long, 0.58 mm inner diameter).
    Faxen: Boolean.
        Apply Faxen’s corrections or not. (Leach,2009,P.R.E)
    d_wall: Float number.
        The distance estimated from beads to cell surface.
        Unit: nm.
        The default value is 380 nm which get from note 20190327. Filament length after shear 60 times.

    Returns
    -------
    Same dimension as f. The torque results after calculation.

    Example:
        GetTorque(84,200,r_bead=495,Faxen=False). This would get about 2202 PN nm. The value is the same as in reference Inoue,2008,JMB.
    '''

    pi = SciCon.pi
    w = 2 * pi * f  # unit: rad/s
    Drag_bead_rotational = 8 * pi * visco * r_bead ** 3
    Drag_bead_translation = 6 * pi * visco * r_bead * re ** 2
    if Faxen == True:
        deno_rotation = 1 - (1 / 8) * (r_bead / d_wall) ** 3
        deno_translation = 1 - (9 / 16) * (r_bead / d_wall) + (1 / 8) * (r_bead / d_wall) ** 3
        Drag_bead = (Drag_bead_rotational / deno_rotation) + (Drag_bead_translation / deno_translation)
        Drag = Drag_bead + Drag_fila
    else:
        Drag_bead = Drag_bead_rotational + Drag_bead_translation
        Drag = Drag_bead + Drag_fila
    Torque = Drag * w

    # print("Drag from beads Translation : ",Drag_bead_translation,"pN nm s/rad")
    # print("Drag from beads rotation : ",Drag_bead_rotational,"pN nm s/rad")
    # print("Total Drag from beads :", Drag_bead,"pN nm s/rad")
    # print("Drag from filament :", Drag_fila,"pN nm s/rad")
    # print("Torque : ", Torque, "PN nm")
    return Torque