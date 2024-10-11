from astropy import __version__ as apversion
from astropy.nddata import CCDData
import numpy as np
import scipy.ndimage as ndi
import ccdproc as ccdp
import matplotlib.pyplot as plt
from astropy.visualization import hist,ZScaleInterval,simple_norm
from astropy.stats import histogram
from astropy.coordinates import ICRS
import scipy.stats as sps
import photutils as phot
from astropy.stats import mad_std
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import subprocess
import re
import os
from glob import glob
from photutils.aperture import CircularAperture
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import pandas as pd

#fitshpath='/home/idies/workspace/Temporary/treed28/scratch/20240903/reduced/'
fitshpath='/home/idies/workspace/Storage/treed28/persistent/bin/'

if not os.path.isdir(fitshpath):
    raise ValueError("Error: Fitsh path does not exist. You need to change this to the correct path for your system (likely /home/idies/workspace/Storage/<username>/persistent/bin)")


def showimage(d, vmin=None, vmax=None, apertures=None,
              cmap='gist_gray', invert=False, xlim=None, ylim=None,
              apcolor='cyan', apwidth=3, apalpha=1,
              figsize=(15,15)):
    '''Show image passed as d, with customizable min and max
    Positional arguments:
        d = image data
    '''
    im = d
    if hasattr(d,'data'):
        im = d.data
    
    med = np.nanmean(im)
    std = sps.median_absolute_deviation(im,axis=None)
    zs = ZScaleInterval()
    if vmin==None or vmax==None:
        vmin,vmax = zs.get_limits(im)
    print(med,std)
    print(vmin,vmax)
    plt.figure(figsize=figsize)
    #mean-2*std
    if invert==True:
        cmap='gist_yarg'
    plt.imshow(im,vmin=vmin,vmax=vmax,cmap=plt.get_cmap(cmap),
               origin='lower')
    
    if not apertures == None:
        apertures.plot(color=apcolor,lw=apwidth,alpha=apalpha)
    if not xlim == None:
        plt.xlim(xlim)
    if not ylim == None:
        plt.ylim(ylim)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    
def imageCenter(imname,hdu,verbose=0):
    """Return the approximate image center's coordinates as a 
    astropy coords object, and as a pair of strings"""
    #Use the image's header information to estimate the position - this 
    #should be good for almost any case
    if verbose>0:
        print("Point RA",hdu.header['OBJCTRA'])
        print("Point Dec",hdu.header['OBJCTDEC'])
    objcoord = SkyCoord(hdu.header['OBJCTRA'],hdu.header['OBJCTDEC'],
                        frame=ICRS,unit=(u.hourangle,u.deg))

    if verbose>0:
        print("objcoord",objcoord)

    radecstr = objcoord.to_string()
    if verbose>0:
        print("radecstr",radecstr)
    rastr,decstr = radecstr.split(" ")
    
    return objcoord,rastr,decstr
    
    
def findStars(imname, hdu, threshold=None, fwhm=4.0, fpnTrim=1024, verbose=0):
    """Find stars in the image for matching to catalogs
    
    Required arguments:
        imname -- string with the image filename
        hdu    -- CCDData or individual astropy.io.fits HDU object
    
    Optional arguments:
        threshold=4.5 -- sigma threshold above background for 
                           DAOStarFinder
        fwhm=4.0        -- Full width half max used by DAOStarFinder
        verbose=0       -- Verbosity of output, use numbers >0 for
                           more output
        fpnTrim=1024  -- Use to trim of the top part of the image by
                         reducing to a number smaller than 1024
                           
    Returns:
        pandas dataframe with star data  
    
    """
    imroot=""
    if isinstance(imname,str):
        imroot = imname[:-4]
    elif hasattr(imname, "as_posix") and callable(getattr(imname, "as_posix")):
        imroot = imname.as_posix()[:-4]
    else:
        raise TypeError('imname argument must be a string or a path object')
    image = hdu.data
    f=hdu.header['FILTER'][-1]

    #Subtract a rough estimate of the sky
    image_m = image - np.median(image)
    bkg_sigma = mad_std(image_m)

    if f=='B' and threshold==None:
        threshold=4.5
    elif threshold==None:
        threshold=4.5

    #Find stars with DAOStarFinder
    daofind = phot.DAOStarFinder(fwhm=fwhm, threshold=threshold*bkg_sigma)
    sources = daofind(image_m)
    
    if fpnTrim>0 and (not sources is None):
        sources = sources[sources['ycentroid']<fpnTrim]

    if verbose>0:
        for col in sources.colnames:
            sources[col].info.format = '%.8g'

        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        aps = CircularAperture(positions,30.0)
        showimage(image,figsize=(7,7),apertures=aps)

    #Get a list ordered by magnitude, and in pandas format
    if sources is None:
        return None
    
    if len(sources)>0:
        source_pd = sources.to_pandas().sort_values('mag')
    
        if verbose>0:
            print(source_pd[['xcentroid','ycentroid','mag']])
    
        #Print the star list to a file
        source_pd.to_csv((imroot + '.stars.csv'),sep=' ',index=False)
    else:
        return None

    
    return source_pd
    
    
def queryGaia(imname, hdu, requeryGaia=True, GaiaROW_LIMIT=2000, 
              searchRadiusScale=1.0, gaiaMagLimit=15.0, verbose=0):
    """Query Gaia database for stars that might be in the image
    
    Required arguments:
        imname -- string with the image filename
        hdu    -- CCDData or individual astropy.io.fits HDU object
    
    Optional arguments:
        requeryGaia=True -- Resend the gaia query
        GaiaROW_LIMIT=2000 -- Maximum number of rows returned from Gaia
        searchRadiusScale=1.5 -- Scaling factor for search radius in 
                                 units of CCD detector size
        verbose=0        -- Verbosity of output, use numbers >0 for
                            more output
    
    Returns:
        pandas dataframe with star data
    """
    

    if isinstance(imname,str):
        imroot = imname[:-4]
    elif hasattr(imname, "as_posix") and callable(getattr(imname, "as_posix")):
        imroot = imname.as_posix()[:-4]
    else:
        raise TypeError('imname argument must be a string or a path object')
        
    f = hdu.header['FILTER'][-1]
    
    objcoord,rastr,decstr = imageCenter(imname,hdu,verbose)


    gmag = 'phot_g_mean_mag'

    if f=='B':
        gmag = 'phot_bp_mean_mag'


    #Grab data for the field from Gaia
    Gaia.ROW_LIMIT = GaiaROW_LIMIT

    #This is a very uncustomizable search, will only return max rows, 
    #and result will be concentrated at center
    #gaia_result = Gaia.query_object_async(coordinate=m29coord, 
    #                                      width=u.Quantity(0.25,u.deg), 
    #                                      height=u.Quantity(0.25,u.deg))

    #Estimate a radius for the Gaia search to enclose the CCD with 
    #some wiggle room
    diagonal = np.sqrt(hdu.header['NAXIS1']**2+hdu.header['NAXIS2']**2)/2.0
    pixsize = hdu.header['XPIXSZ'] * u.um
    focallength = hdu.header['FOCALLEN'] * u.mm
    #diameter = hdu.header['APTDIA'] * u.mm
    pixscale = 206265.0*u.arcsec/focallength * pixsize #arcsec/pixel
    searchRadius = searchRadiusScale * diagonal * pixscale.to(u.deg)
    if verbose>0:
        print("Search radius =",searchRadius)
    elif verbose>1:
        print("Diagonal (pix)",diagonal)
        print("Pixel size",pixsize)
        print("Focal length",focallength)
        print("Pixel scale",pixscale)
        print("Search radius",searchRadius)
        

    if requeryGaia or not os.path.exists(imroot + '.gaia.csv'):
        
        if verbose>=2:
            print("""Workaround instructions:
            Easy option if you've already plate solved an image for this field before:
            1. Copy the <previous_image_name>.gaia.csv file from the previous plate solve to <current_image_name>.gaia.csv
            2. Run the plate solve again with requeryGaia=False
              
            Harder option, for when you've never plate solved this region of sky before
            1. Go to https://gea.esac.esa.int/archive/
            2. Click 'Search'
            3. Click the 'Advanced (ADQL)' tab
            4. Copy and paste the query printed below into the top text box then click 'Submit Query'
            5. At the bottom of the page, change the download format to CSV while waiting for the query to complete running
            6. Once a check mark appears, download the results with one of the icons on the right hand side of the job listing
            7. Upload the file to the sciserver directory with the reduced images
            8. In a terminal Run the command "perl -pne 's/,/ /g' <...-result.csv> > <image_name>.gaia.csv"
            9. Run the plate solve again with requeryGaia=False
              
            """)
        
        if verbose>=1:
            print("""SELECT DISTANCE(POINT('ICRS',ra,dec),
                              POINT('ICRS',%s,%s))
                              AS dist,ra,dec,phot_g_mean_mag,
                              phot_bp_mean_mag,phot_rp_mean_mag
                              FROM gaiaedr3.gaia_source
                              WHERE 1=CONTAINS(
                              POINT('ICRS',ra,dec),
                              CIRCLE('ICRS',%s,%s,%g))
                              AND %s < %g
                              ORDER BY %s ASC""" % (rastr, decstr,
                                                    rastr, decstr,
                                                    searchRadius.value, 
                                                    gmag, gaiaMagLimit,
                                                    gmag
                                                   ))

        job = Gaia.launch_job("""SELECT DISTANCE(POINT('ICRS',ra,dec),
                              POINT('ICRS',%s,%s))
                              AS dist,ra,dec,phot_g_mean_mag,
                              phot_bp_mean_mag,phot_rp_mean_mag
                              FROM gaiaedr3.gaia_source
                              WHERE 1=CONTAINS(
                              POINT('ICRS',ra,dec),
                              CIRCLE('ICRS',%s,%s,%g))
                              AND %s < %g
                              ORDER BY %s ASC""" % (rastr, decstr,
                                                    rastr, decstr,
                                                    searchRadius.value, 
                                                    gmag, gaiaMagLimit,
                                                    gmag
                                                   ))

        gaia_result = job.get_results()

        #Write to a file
        gaia_pd = gaia_result['ra','dec',gmag].to_pandas()
        gaia_pd.sort_values(gmag)
        gaia_pd.to_csv(imroot + '.gaia.csv',sep=' ',index=False)
    
        if verbose>0:
            print(gaia_result['ra','dec',gmag])
    else:
        gaia_pd = pd.read_csv(imroot + '.gaia.csv',sep='\s+')
        
    return gaia_pd
        
def grmatch_help():
    ret = subprocess.check_call([fitshpath + '/grmatch','--long-help'])
    print(ret)
    
def grtrans_help():
    subprocess.check_call([fitshpath + '/grtrans','--long-help'])
    
def fiheader_help():
    subprocess.check_call([fitshpath + '/fiheader','--long-help'])
    
def checkSolve(imname,hdu,verbose=0):
    
    imroot=""
    if isinstance(imname,str):
        imroot = imname[:-4]
    elif hasattr(imname, "as_posix") and callable(getattr(imname, "as_posix")):
        imroot = imname.as_posix()[:-4]
    else:
        raise TypeError('imname argument must be a string or a path object')
        
    tfile = imroot + '.match.transform'
    dxfit = []
    dyfit = []
    matchline=""
    lines = []
    
    
    with open(tfile,'r') as file:
        lines = file.readlines()
        
        for line in lines:
            words = line.rstrip().split(" ")
            #print(words[0])
            if words[0] == 'dxfit=':
                #print(''.join(words[1:]).split(','))
                dxfit = np.array(''.join(words[1:]).split(',')).astype(np.float)
            if words[0] == 'dyfit=':
                dyfit = np.array(''.join(words[1:]).split(',')).astype(np.float)
    
    dxfit[1:3] = dxfit[1:3] / 3600.0
    dyfit[1:3] = dyfit[1:3] / 3600.0
    
    if verbose>0:
        print("dxfit",dxfit)
        print("dyfit",dyfit)
    
    anglex = np.arctan2(-dxfit[2],dxfit[1]) #* 180.0/np.pi
    angley = np.arctan2(dyfit[1],dyfit[2]) #* 180.0/np.pi
    angle = 0.5*(anglex+angley)
    
    scalex = 1.0/(dxfit[1]/np.cos(anglex))
    scaley = 1.0/(dyfit[2]/np.cos(angley))
    
    fracanglediff = np.abs((anglex - angley)/anglex)
    fracscalediff = np.abs((scalex-scaley)/scalex)
    anglediff = np.abs(anglex-angley)
    
    if anglediff<0.1 and fracscalediff<0.03:
        retval = True
    else:  
        retval = False
        
    if verbose>0:
        print("")
        print("scalex, scaley:",scalex,scaley)
        print("anglex, angley:",anglex*180/np.pi,angley*180/np.pi)
        print("fracscalediff:",fracscalediff)
        print("anglediff:",anglediff)
        print("")
        print(''.join(lines))
        print(retval)
    
    return retval
    
        
def plateSolve(imname, hdu, grmatch_nstars = None, grmatch_level=None, 
               grmatch_crm='conformable', grmatch_maxdist=2.0,
               grmatch_order=1, manualSelect=False, verbose=0):
    """Solve for the WCS using external tool fitsh's grmatch and
    grtrans programs. Some arguments for these programs can be 
    adjusted.
    
    Required arguments:
        imname -- string with the image filename
        hdu    -- CCDData or individual astropy.io.fits HDU object
        
    Optional arguments:
        grmatch_nstars=None -- maximum number of stars used for match,
                               defaults are 20 if filter is B, 50 
                               otherwise
        grmatch_level=None  -- grmatch triangulation level. Options are
                               'level=<N>' or 'full'
        grmatch_crm='conformable' -- is the handedness of the pixel-grid
                                     the same as the sky coordinates. If
                                     yes use 'conformable', if no use
                                     'reverse', if unknown use 'mixed'
        grmatch_maxdist=2.0 -- maximum distance for match in arcsec
        grmatch_order=1     -- polynomial order of astrometric fit 
                               transformation
        manualSelect=False  -- Use a manually selected list of matches
    
    """
    
    if isinstance(imname,str):
        imroot = imname[:-4]
    elif hasattr(imname, "as_posix") and callable(getattr(imname, "as_posix")):
        imroot = imname.as_posix()[:-4]
    else:
        raise TypeError('imname argument must be a string or a path object')
    
    f = hdu.header['FILTER'][-1]
    
    
    listSuffix='.csv'
    if manualSelect:
        listSuffix='.sel.csv'
    
    objcoord,rastr,decstr = imageCenter(imname,hdu,verbose)

    #Project the Gaia coordinates onto a plane tangent to a reference point
    grtrans1 = [fitshpath + 'grtrans', '--input', imroot + '.gaia' + listSuffix, 
                '--wcs', 'tan,ra=' + rastr + ',dec=' + decstr + ',degrees',
                '--col-radec', '1,2', '--col-out', '4,5', '--output',
                imroot + '.gaia.proj']

    #Parse the grmatch options
    if grmatch_nstars==None:
        grmatch_nstars=100
        if f=='B':
            grmatch_nstars=20
            
    if grmatch_level==None:
        grmatch_level = 'full'
        if f=='B':
            grmatch_level = 'full'
         
    regexp1 = re.compile(r'^%s' % (grmatch_level))
    regexp2 = re.compile(r'^level=\d')
    if (not regexp1.search('full')) and (not regexp2.search(grmatch_level)):
         raise ValueError("""The grmatch_level argument's value (%s) is 
                          invalid. Valid options are 'full' and 'level=<N>',
                          where <N> is an integerr between 1 and 4. See
                          grmatch documentation for details.""")
            
    grmatch_maxdist = float(grmatch_maxdist)
    grmatch_order = int(grmatch_order)
    
    #Build the grmatch, grtrans, and fiheader commands

    #Cross match detected stars and gaia stars, and fit polynomials to 
    #transform one to the other
    grmatch = [fitshpath + 'grmatch', '-r', imroot + '.gaia.proj', 
               '-i', imroot + '.stars' + listSuffix,
               '-o', imroot + '.match.csv', 
               '--match-points', '--col-ref', '4,5', '--col-inp', '2,3', 
               '--output-transformation', imroot + '.match.transform', 
               '--col-ref-ordering', '-3', '--col-inp-ordering', '-11',
               '--order', str(grmatch_order), 
               '--triangulation', 
               grmatch_level + ',unitarity=0.001,' + 
               grmatch_crm + ',maxnumber=' + str(grmatch_nstars),
               '--max-distance', str(grmatch_maxdist), 
               '--weight', 'reference,column=3,magnitude,power=1',
               '--comment']

    #Transform the projected Gaia coordinates to image coordinates
    grtrans2 = [fitshpath + 'grtrans', 
                '--input', imroot + '.gaia.proj', 
                '--col-xy', '4,5', '--col-out', '6,7', 
                '--input-transformation', imroot + '.match.transform',
                '--output', imroot + '.gaia.proj.plate']

    #Fit a WCS solution to the transformed Gaia coordinates    
    grtrans3 = [fitshpath + 'grtrans', 
                '--input', imroot + '.gaia.proj.plate', 
                '--col-ref', '1,2', '--col-fit', '6,7',
                '--wcs', 'tan,order=' + str(grmatch_order) + ',ra=' + rastr + ',dec=' + decstr,
                '>', imroot + '.wcs']

    #Add the WCS information to the image header
    fiheader = [fitshpath + 'fiheader', 
                '-i', imroot + '.fit', 
                '-o', imroot + '.wcs.fit', 
                '--update', 
                """$(awk '{printf("%%s",$0)}' %s.wcs)""" % (imroot)]

    #fiheader = [fitshpath + 'sethead','-n',imname,'@' + imroot + '.wcs']


    #Run the commands

    if verbose>0: print(' '.join(grtrans1) + '\n')
    rv1 = subprocess.check_call(grtrans1)
    if verbose>0: print("Return code:",rv1,'\n\n')

    if verbose>0: print(' '.join(grmatch) + '\n')
    try:
        rv2 = subprocess.check_call(grmatch)
    except:
        print("Error: grmatch command failed")
    if verbose>0:
        try:
            with open(imroot + '.match.transform','r') as tfile:
                print(tfile.read())
            if verbose>0: print("Return code:",rv2,'\n\n')
        except:
            print("Couldn't open", 
                  imroot + '.match.transform. grmatch probably failed')
            
    if checkSolve(imname, hdu, verbose)==False:
        print("Error in plate solve, stopping before creating wcs solved image.")
        return False
    

    if verbose>0: print(' '.join(grtrans2) + '\n')
    rv3 = subprocess.check_call(grtrans2)
    if verbose>0: print("Return code:",rv3,'\n\n')

    if verbose>0: print(' '.join(grtrans3) + '\n')
    rv4 = os.system(" ".join(grtrans3))
    if verbose>0: print("Return code:",rv4,'\n\n')

    if verbose>0: print(' '.join(fiheader) + '\n')
    rv5 = os.system(" ".join(fiheader))
    if verbose>0: print("Return code:",rv5,'\n\n')
        
    return True


def manualSolvePlot(imname,hdu,rotateGaia=-7.43):
    #Plot the stars for diagnostic purposes
    
    imroot=""
    if isinstance(imname,str):
        imroot = imname[:-4]
    elif hasattr(imname, "as_posix") and callable(getattr(imname, "as_posix")):
        imroot = imname.as_posix()[:-4]
    else:
        raise TypeError('imname argument must be a string or a path object')

    f = hdu.header['FILTER'][-1]

    gmag = 'phot_g_mean_mag'
    if f=='B':
        gmag = 'phot_bp_mean_mag'

    
    mpld3.enable_notebook()

    objcoord,rastr,decstr = imageCenter(imname,hdu)

    ra = float(rastr)
    dec = float(decstr)
    
    starsname = imroot + '.'

    s = pd.read_csv(imroot + '.stars.csv',sep='\s+')
    g = pd.read_csv(imroot + '.gaia.proj', sep='\s+', header=None)
    
    rotrad = rotateGaia*np.pi/180.0
    grotx = g.iloc[:,-2] * np.cos(rotrad) - g.iloc[:,-1] * np.sin(rotrad)
    groty = g.iloc[:,-2] * np.sin(rotrad) + g.iloc[:,-1] * np.cos(rotrad)
    #grotx = g['ra'] * np.cos(rotrad) - g['dec'] * np.sin(rotrad)
    #groty = g['ra'] * np.sin(rotrad) + g['dec'] * np.cos(rotrad)
    
    positions = np.transpose((s['xcentroid'], s['ycentroid']))
    aps = CircularAperture(positions,30.0)
    showimage(hdu,figsize=(7,7),apertures=aps)


    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    #scatter1 = ax[0].scatter(60*(g['ra']-ra),60*(g['dec']-dec),
    #                      s=(20-g[gmag])**2,c='black')
    scatter1 = ax[0].scatter(grotx*60,groty*60,
                          s=(20-g.iloc[:,-3])**2,c='black')
    #scatter1 = ax[0].scatter(grotx*60,groty*60,
    #                      s=(20-g[gmag])**2,c='black')
    scatter2 = ax[1].scatter((s['xcentroid']-490)*0.65/60.0,
                          (s['ycentroid']-490)*0.65/60.0,
                          s=(20-(s['mag']+14.35))**2,c='red')
    #ax[0].set_aspect(1.0/np.cos(dec*np.pi/180.0))
    ax[0].set_aspect(1)
    ax[0].set_xlabel("Black=Gaia")
    ax[1].set_aspect(1)
    ax[1].set_xlabel("Red=Image")
    
    ax0xlim = np.asarray(ax[0].get_xlim())
    ax0ylim = np.asarray(ax[0].get_ylim())
    ax1xlim = np.asarray(ax[1].get_xlim())
    ax1ylim = np.asarray(ax[1].get_ylim())
    
    xlim = [np.amin(np.array([ax0xlim[0],ax1xlim[0]])),np.amax(np.array([ax0xlim[1],ax1xlim[1]]))]
    ylim = [np.amin(np.array([ax0ylim[0],ax1ylim[0]])),np.amax(np.array([ax0ylim[1],ax1ylim[1]]))]
    
    ax[0].set_xlim(xlim[0],xlim[1])
    ax[0].set_ylim(ylim[0],ylim[1])
    ax[1].set_xlim(xlim[0],xlim[1])
    ax[1].set_ylim(ylim[0],ylim[1])

    labels1 = ['{0}'.format(i) for i in g.index]
    labels2 = ['{0}'.format(i) for i in s.index]
    tooltip1 = mpld3.plugins.PointLabelTooltip(scatter1, labels=labels1)
    tooltip2 = mpld3.plugins.PointLabelTooltip(scatter2, labels=labels2)
    mpld3.plugins.connect(fig, tooltip1)
    mpld3.plugins.connect(fig, tooltip2)
    
    
def autoSolve(imname,hdu, 
              threshold=None, fwhm=4.0, fpnTrim=1024,
              requeryGaia=True, GaiaROW_LIMIT=2000, 
              searchRadiusScale=1.0, gaiaMagLimit=15.0,
              grmatch_nstars = None, grmatch_level=None, 
              grmatch_crm='conformable', grmatch_maxdist=2.0,
              grmatch_order=1, manualSelect=False,
              returnDiagnostics=False,verbose=0):
    
    """Attempt to automatically solve for the WCS of an image
    by:
       1. Finding stars in the image
       2. Downloading a reference catalog of stars from Gaia
       3. Crossmatching the stars in the image with Gaia's stars
       4. Solving for the plate equations
       5. Checking that the solution is sensible for the HRPO instrument.
    Solve for the WCS using external tool fitsh's grmatch and
    grtrans programs. Some arguments for these programs can be 
    adjusted.
    
    Required arguments:
        imname -- string with the image filename
        hdu    -- CCDData or individual astropy.io.fits HDU object
        
        
    Optional arguments:
        verbose=0     --verbosity of output
        returnDiagnostics=False  --Return catalogs
        
        
    Optional arguments affecting star finding:
    
        threshold=4.5 -- sigma threshold above background for 
                           DAOStarFinder
        fwhm=4.0      -- Full width half max used by DAOStarFinder
        verbose=0     -- Verbosity of output, use numbers >0 for
                           more output
        fpnTrim=1024  -- Use to trim of the top part of the image by
                         reducing to a number smaller than 1024
      
      
    Optional arguments affecting the reference catalog:
    
        requeryGaia=True -- Resend the gaia query
        GaiaROW_LIMIT=2000 -- Maximum number of rows returned from Gaia
        searchRadiusScale=1.5 -- Scaling factor for search radius in 
                                 units of CCD detector size
        verbose=0        -- Verbosity of output, use numbers >0 for
                            more output  
      
      
    Optional arguments affecting the cross match and plate solve:
    
        grmatch_nstars=None -- maximum number of stars used for match,
                               defaults are 20 if filter is B, 50 
                               otherwise
        grmatch_level=None  -- grmatch triangulation level. Options are
                               'level=<N>' or 'full'
        grmatch_crm='conformable' -- is the handedness of the pixel-grid
                                     the same as the sky coordinates. If
                                     yes use 'conformable', if no use
                                     'reverse', if unknown use 'mixed'
        grmatch_maxdist=2.0 -- maximum distance for match in arcsec
        grmatch_order=1     -- polynomial order of astrometric fit 
                               transformation
        manualSelect=False  -- Use a manually selected list of matches
    
    
    Returns:
        Boolean -- True if the check indicates the solution was successful,
                   False otherwise. 
        or, if returnDiagnostics is true:
        Boolean check, (pandas.DataFrame stars, pandas.DataFrame gaia)
        
    """
    
    stars = findStars(imname,hdu,threshold=threshold, fwhm=fwhm, fpnTrim=fpnTrim,
                      verbose=verbose)
    if stars is None:
        check=False
        if verbose>0:
            print("No stars found in image")
        if returnDiagnostics==True:
            return check, (stars,None)
        return check
    gaia = queryGaia(imname,hdu,requeryGaia=requeryGaia, GaiaROW_LIMIT=GaiaROW_LIMIT, 
                     searchRadiusScale=searchRadiusScale, gaiaMagLimit=gaiaMagLimit,
                     verbose=verbose)
    plateSolve(imname,hdu,grmatch_nstars=grmatch_nstars, grmatch_level=grmatch_level,
               grmatch_crm=grmatch_crm, grmatch_maxdist=grmatch_maxdist,
               grmatch_order=grmatch_order, verbose=verbose)
    check = checkSolve(imname,hdu, verbose=verbose)
    
    if verbose>0:
        print(imname,check)
        
    if returnDiagnostics==True:
        return check, (stars, gaia)
    return check

def manualSolve(imname,hdu,gaiaStars,imageStars,
                grmatch_nstars = None, grmatch_level='full', 
                grmatch_crm='conformable', grmatch_maxdist=2.0,
                grmatch_order=1, verbose=0):
    
    """Manually cross-match image and reference stars for cases where
    automatic plate solving fails. Stars can be identified by calling
    plateSolve.manualSolvePlot(imname,hdu).
    
    Required arguments:
        imname -- string with the image filename
        hdu    -- CCDData or individual astropy.io.fits HDU object
        gaiaStars -- list of integers containing star IDs from the gaia list
        imageStars - list of integers containing star IDs from the image list
        
    Optional arguments affecting the cross match and plate solve:
    
        grmatch_nstars=None -- maximum number of stars used for match,
                               defaults are 20 if filter is B, 50 
                               otherwise
        grmatch_level='full'  -- grmatch triangulation level. Options are
                               'level=<N>' or 'full'
        grmatch_crm='conformable' -- is the handedness of the pixel-grid
                                     the same as the sky coordinates. If
                                     yes use 'conformable', if no use
                                     'reverse', if unknown use 'mixed'
        grmatch_maxdist=2.0 -- maximum distance for match in arcsec
        grmatch_order=1     -- polynomial order of astrometric fit 
                               transformation
        manualSelect=False  -- Use a manually selected list of matches    
    
    """
    
    imroot=""
    if isinstance(imname,str):
        imroot = imname[:-4]
    elif hasattr(imname, "as_posix") and callable(getattr(imname, "as_posix")):
        imroot = imname.as_posix()[:-4]
    else:
        raise TypeError('imname argument must be a string or a path object')
    
    #Select the stars from the lists and store them in new csv files
    gaia = pd.read_csv(imroot + '.gaia.csv',sep='\s+')
    gaia_sel = gaia[gaia.index.isin(gaiaStars)]
    gaia_sel.to_csv(imroot + '.gaia.sel.csv',index=False,sep=' ')
    
    stars = pd.read_csv(imroot + '.stars.csv',sep='\s+')
    imag_sel = stars[stars.index.isin(imageStars)]
    imag_sel.to_csv(imroot + '.stars.sel.csv',index=False,sep=' ')
    
    #Tell platesolve to use the 
    plateSolve(imname,hdu,manualSelect=True,
               grmatch_nstars=grmatch_nstars, grmatch_level=grmatch_level,
               grmatch_crm=grmatch_crm, grmatch_maxdist=grmatch_maxdist,
               grmatch_order=grmatch_order)
    
    check = checkSolve(imname,hdu,verbose=verbose)
    
    if verbose>0:
        print(imname,check)
        
    return check
    
    
    
def cleanup(imname,check=True):
    
    """Delete the extraneous files created by the plateSolve process.
    Run with check=False to actually delete them. If check is true it will
    just list the files to be deleted.
    
    Required arguments:
        imname -- string or path object
    
    Optional arguments:
        check=True -- boolean. if True, do not delete files, just display message. If 
                      False, delete the files.
    """
    
    imroot=""
    if isinstance(imname,str):
        imroot = imname[:-4]
    elif hasattr(imname, "as_posix") and callable(getattr(imname, "as_posix")):
        imroot = imname.as_posix()[:-4]
    else:
        raise TypeError('imname argument must be a string or a path object')
        
    print("Deleting the following files:")
    files_to_delete = list((set(glob(imroot + ".stars*")) | set(glob(imroot + ".gaia*")) | set(glob(imroot + ".wcs*")) | set(glob(imroot + ".match*"))) - set(glob(imroot + ".wcs.fit")))
    print(files_to_delete)
    
    if check==True:
        print("If you want to delete the files, rerun this command with check=False")
    else:
        for f in files_to_delete:
            if os.path.isfile(f):
                os.remove(f)
