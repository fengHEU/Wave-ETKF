"""
Sparse reconstruction processing functions from WASSFAST 
(https://gitlab.com/fibe/wassfast)
Modified to integrate with ETKF or KF measurement pipeline.

Key modifications:
- Combined multiple processing steps into a unified measurement pipeline
- Retained core reconstruction algorithms from WASSFAST

Original paper citation:
Bergamasco, F., Benetazzo, A., Yoo, J., Torsello, A., Barbariol, F., Jeong, J. Y., ... & Cavaleri, L. (2021). 
Toward real-time optical estimation of ocean waves' space-time fields. 
Computers & Geosciences, 147, 104666.

Original copyright (C) 2020 Filippo Bergamasco
Released under GNU General Public License v3.0
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import signal



def sparse_elevation(I0, I1, config, processors, DEBUGDIR, settings, 
                     reduce_mode=False, visualize_mask=False):
    cam0mask = None
    cam1mask = None

    p3dN, cam0mask, cam1mask = estimate_scattered_point_cloud(I0, I1, cam0mask, cam1mask, 
                                                              config, processors, DEBUGDIR, 
                                                              settings, idx=0)

    scalefacx = config.limits.xmax - config.limits.xmin
    scalefacy = config.limits.ymax - config.limits.ymin
    p3dN_norm = np.copy(p3dN)
    p3dN_norm[0, :] = (p3dN_norm[0, :] - config.limits.xmin) / scalefacx - 0.5
    p3dN_norm[1, :] = (p3dN_norm[1, :] - config.limits.ymin) / scalefacy - 0.5
    pt2 = p3dN_norm[0:2, :].astype(np.float32).T
    npts = pt2.shape[0]
    keep = np.ones((npts, 1), dtype=np.bool_)

    if reduce_mode:
        p3dN = reuce_points(p3dN, config, processors, settings, pt2, npts, keep)
        p3dN_norm = p3dN_norm[:,np.squeeze(keep) ]
        pt2 = p3dN_norm[0:2, :].astype(np.float32).T

    #start_time = time.time()
    [XX, YY] = np.meshgrid(np.linspace(config.limits.xmin, config.limits.xmax, config.N),
                           np.linspace(config.limits.ymin, config.limits.ymax, config.N))
    ZI = griddata(p3dN[0:2, :].T, p3dN[2, :], (XX, YY), method='linear')
    #print(f'griddata time{time.time()-start_time}')

    # Create a mask to mark which grid points are original data and which are interpolated
    mask_original = np.zeros_like(ZI, dtype=bool)
    for i in range(pt2.shape[0]):
        x, y = pt2[i]
        x_idx = int((x + 0.5) * config.N)
        y_idx = int((y + 0.5) * config.N)
        x_idx = np.clip(x_idx, 0, config.N - 1)
        y_idx = np.clip(y_idx, 0, config.N - 1)
        mask_original[y_idx, x_idx] = True

    mask_interpolated = ~mask_original
    mask = compute_mask(ZI)
    ZI[np.isnan(ZI)] = 0
    ZI = ZI * mask

    if visualize_mask:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_original, cmap='gray')
        plt.title('Mask of Original Data')
        # 显示ZI
        plt.figure(figsize=(10, 10))
        plt.imshow(ZI, cmap='jet')
        plt.title('Elevation Map')
        plt.show()

    return ZI, mask_original, mask_interpolated


def tukeywin_m( NN, Alpha ):
    """
    Implements and return a Matlab-stlye tukeywin
    """
    w = signal.windows.tukey(NN,Alpha )
    w = w.reshape(NN,1)*w
    return w.astype( np.float64 )

def compute_mask(ZI, tukey_p=0.08, gauss_sigma=2.0 ):
    assert( ZI.shape[0] == ZI.shape[1] )
    N = ZI.shape[0]
    mask = np.logical_not( np.isnan( ZI ) ).astype( np.float32 )
    mask = cv.GaussianBlur( mask, (0,0), sigmaX=gauss_sigma, sigmaY=gauss_sigma )
    mask[ mask<0.99 ] = 0
    mask = cv.GaussianBlur( mask, (0,0), sigmaX=gauss_sigma, sigmaY=gauss_sigma )
    maskborder = tukeywin_m( N, tukey_p )
    mask = mask * maskborder
    return mask

# Debug extracted features in plane space
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
 
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
 
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
 
            cv.line(img, pt1, pt2, delaunay_color, 1)
            cv.line(img, pt2, pt3, delaunay_color, 1)
            cv.line(img, pt3, pt1, delaunay_color, 1)

# Draw delaunay triangles
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def debug_featuresP( matcher, I0p, I1p, outdir="dbg/", cam0name="cam0P_features", cam1name="cam1P_features", image_idx=0 ):
    I0p_aux = cv.cvtColor( I0p, cv.COLOR_GRAY2BGR )

    for idx in range(0,matcher.features_0P_all.shape[0]):
        #__import__("IPython").embed()
        cv.drawMarker(I0p_aux, tuple( matcher.features_0P_all[idx,:].astype(int) ), (0,0,0), markerType=cv.MARKER_TILTED_CROSS, markerSize=5  )

    subdiv = cv.Subdiv2D( (0,0,I0p_aux.shape[1],I0p_aux.shape[0]) )

    for idx in range(0,matcher.features_0P.shape[0]):
        p = tuple( matcher.features_0P[idx,:].astype(int) )
        subdiv.insert( (int(p[0]), int(p[1]) ) ) 
        cv.drawMarker(I0p_aux, p, (0,0,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5  )

    draw_delaunay( I0p_aux, subdiv, (0,0,255) ) 
    cv.imwrite("%s/%06d_%s.jpg"%(outdir,image_idx,cam0name), I0p_aux)

    I1p_aux = cv.cvtColor( I1p, cv.COLOR_GRAY2BGR )
    f1_int = np.round(matcher.features_1P)

    subdiv = cv.Subdiv2D( (0,0,I0p_aux.shape[1],I0p_aux.shape[0]) )

    for idx in range(0,f1_int.shape[0]):
        p = tuple( f1_int[idx,:].astype(int) )
        subdiv.insert( (int(p[0]), int(p[1]) ) ) 
        cv.drawMarker(I1p_aux, p, (0,0,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5  )

    draw_delaunay( I1p_aux, subdiv, (0,0,255) ) 
    cv.imwrite("%s/%06d_%s.jpg"%(outdir,image_idx,cam1name), I1p_aux)


def create_mask( I0, I1, config ):
    cam0mask = config.cam.c2sH_cam0.warp(np.ones_like(I0))
    cam1mask = config.cam.c2sH_cam1.warp(np.ones_like(I1))
    # Set 1px border on image boundary
    cam0mask[0,:]=0 ; cam0mask[-1,:]=0 ; cam0mask[:,0]=0 ; cam0mask[:,-1]=0
    cam1mask[0,:]=0 ; cam1mask[-1,:]=0 ; cam1mask[:,0]=0 ; cam1mask[:,-1]=0

    cam0mask = cv.erode( cam0mask, np.ones((7,7)))
    cam1mask = cv.erode( cam1mask, np.ones((7,7)))
    return cam0mask, cam1mask

def estimate_scattered_point_cloud( I0, I1, cam0mask, cam1mask, config, processors, DEBUGDIR, settings, 
                                   idx=0, debug_mode=False ):
    #start_time = time.time()
    I0 = cv.undistort(I0, config.cam.intr_00, config.cam.dist_00 )
    I1 = cv.undistort(I1, config.cam.intr_01, config.cam.dist_01 )
    #print(f'undistort time{time.time()-start_time}')
    if cam0mask is None:
        cam0mask, cam1mask = create_mask(I0,I1,config)
    I0p = config.cam.c2sH_cam0.warp( I0 )
    I1p = config.cam.c2sH_cam1.warp( I1 )

    processors.clahe1 = cv.createCLAHE(clipLimit=settings.getfloat("ImgProc", "I0_cliplimit"),
                                       tileGridSize=(settings.getint("ImgProc", "I0_gridsize"),
                                                     settings.getint("ImgProc", "I0_gridsize")))
    processors.clahe2 = cv.createCLAHE(clipLimit=settings.getfloat("ImgProc", "I1_cliplimit"),
                                       tileGridSize=(settings.getint("ImgProc", "I1_gridsize"),
                                                     settings.getint("ImgProc", "I1_gridsize")))
    I0p = processors.clahe1.apply(I0p)
    I1p = processors.clahe2.apply(I1p)
    processors.clahe1 = None
    processors.clahe2 = None
    

    if debug_mode:
        cv.imwrite("%s/%06d_cam0P.jpg" % (DEBUGDIR, idx), I0p)
        cv.imwrite("%s/%06d_cam1P.jpg" % (DEBUGDIR, idx), I1p)
        cv.imwrite("%s/cam0mask.jpg" % DEBUGDIR, cam0mask * 255)
        cv.imwrite("%s/cam1mask.jpg" % DEBUGDIR, cam1mask * 255)

    #start_time = time.time()
    processors.matcher.extract_features( I0p,
                                         I1p,
                                         cam0mask,
                                         cam1mask,
                                         fb_threshold = settings.getfloat("Flow","fb_threshold"),
                                         winsize=(settings.getint("Flow","winsize"),
                                         settings.getint("Flow","winsize") ),
                                         maxlevel=settings.getint("Flow","maxlevel"),
                                         optflo_method=settings.get("Flow","method") )
    #print(f'extract_features time{time.time()-start_time}')

    if debug_mode:
        debug_featuresP(processors.matcher, I0p, I1p, outdir=DEBUGDIR, image_idx=idx)

    features_0 = config.cam.c2sH_cam0.transform( processors.matcher.features_0P.T, inverse=True )
    features_1 = config.cam.c2sH_cam1.transform( processors.matcher.features_1P.T, inverse=True )

    #start_time = time.time()
    p3d = cv.triangulatePoints( config.cam.P0cam, config.cam.P1cam, features_0, features_1 )
    p3d = p3d[0:3,:] / p3d[3,:]
    #print(f'triangulatePoints time{time.time()-start_time}')

    p3dN = np.matmul( config.cam.Rpl, p3d ) + config.cam.Tpl
    p3dN = p3dN * np.array( [ [config.baseline], [config.baseline], [-config.baseline]]  )

    quantile_level = 0.99  # 0.8
    if quantile_level>0.0 and quantile_level<=1.0:  
        abselevations = np.abs(p3dN[2,:])  
        zquant = np.quantile(abselevations, quantile_level )
        good_pts = abselevations < zquant
        p3dN = p3dN[:, good_pts]

    if debug_mode:
        plt.scatter(p3dN[0, :], p3dN[1, :], s=1, c=p3dN[2, :], vmin=config.limits.zmin, vmax=config.limits.zmax)
        plt.axis('equal')
        plt.grid()
        plt.gca().invert_yaxis()
        plt.savefig('%s/%06d_scatter.png' % (DEBUGDIR, idx))
        plt.close()
    return p3dN, cam0mask, cam1mask