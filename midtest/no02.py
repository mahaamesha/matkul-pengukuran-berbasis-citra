# midtest no 2
# name  : Avima Haamesha
# nim   : 10219084

# i convert matlab program
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt


# set default size for matplotlib
plt.rcParams.update({'font.size': 10})

def get_PSF(PSF:int):
    """ create PSF, model of camera's aperture (bukaan kamera)\\
        return the PSF model and 2D plane as meshgrid\\
        
        PSF : int   the size of PSF\\
    """
    f = np.zeros((256, 256))     # create 2D matrix, all value 0
    # make a 2D plane (u x v)
    u, v = np.meshgrid(np.arange(-f.shape[0]//2, f.shape[0]//2), 
                        np.arange(-f.shape[1]//2, f.shape[1]//2), indexing='ij')
    H = u**2 + v**2 < PSF**2    # make circle, ilustrate the PSF
    return H, u, v


def get_OTF_MTF(H):
    """ OTF (optical transfer function) is fourier transform of PSF H\\
        MTF (modulation transfer function) is the absolute value of the fourier transform of the LSF\\
        
        H   : Any  PSF H\\
    """
    # compute 2D Fourier transform of the PSF H
    F = fft2(H)     # OTF
    # shift zero-frequency component to center of spectrum
    FF = fftshift(np.abs(F))    # np.abs(F) or FF is MTF
    return F, FF


def plot_PSF(u, v, H):
    """ plot single PSF in 2d and 3d spatial domain\\
        
        u   : Any   2d matrix for x line\\
        v   : Any   2d matrix for y line\\
        H   : Any   PSF H\\
    """
    fig = plt.figure(figsize=(10,5)); plt.suptitle('PSF', fontsize='x-large')

    ax1 = fig.add_subplot(121)
    ax1.imshow(H, cmap='gray')
    ax1.set_title('PSF in 2D Spatial Frequency Domain')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(u, v, H, cmap='viridis')
    ax2.set_title('PSF in 3D Spatial Frequency Domain')
    ax2.set_xlabel('freq x')
    ax2.set_ylabel('freq y')
    ax2.set_zlabel('magnitude')
    
    return fig


def plot_OTF_complex(u, v, F):
    """ represent 2d mapping for real and imaginer OTF\\
        3d (@1 plot) and 2d (@2 plots) in spatial frequency domain for each component\\
        so i have 2x3 plots in total\\
        
        u   : Any   2d matrix for x line\\
        v   : Any   2d matrix for y line\\
        H   : Any   PSF H\\
    """
    fig = plt.figure(figsize=(12,6)); plt.suptitle('OTF', fontsize='x-large')
    
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_surface(u, v, np.real(F), cmap='viridis')
    ax1.set_title('OTF Real')
    ax1.set_xlabel('freq x')
    ax1.set_ylabel('freq y')
    ax1.set_zlabel('magnitude')
    
    ax2 = fig.add_subplot(232)
    ax2.plot(u, np.real(F))
    ax2.set_title('OTF Real vs Freq x')
    ax2.set_xlabel('freq x')
    ax2.set_ylabel('magnitude')

    ax3 = fig.add_subplot(233)
    ax3.plot(v, np.real(F))
    ax3.set_title('OTF Real vs Freq y')
    ax3.set_xlabel('freq y')
    ax3.set_ylabel('magnitude')

    ax4 = fig.add_subplot(234, projection='3d')
    ax4.plot_surface(u, v, np.imag(F), cmap='viridis')
    ax4.set_title('OTF Imaginer')
    ax4.set_xlabel('freq x')
    ax4.set_ylabel('freq y')
    ax4.set_zlabel('magnitude')

    ax5 = fig.add_subplot(235)
    ax5.plot(u, np.imag(F))
    ax5.set_title('OTF Imaginer vs Freq x')
    ax5.set_xlabel('freq x')
    ax5.set_ylabel('magnitude')

    ax6 = fig.add_subplot(236)
    ax6.plot(v, np.imag(F))
    ax6.set_title('OTF Imaginer vs Freq y')
    ax6.set_xlabel('freq y')
    ax6.set_ylabel('magnitude')
    
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    return fig


def plot_MTF(u, v, FF):
    """ represent MTF (for single PSF) in 3d frequency domain

        FF  : Any   MTF FF\\
    """
    fig = plt.figure(); plt.suptitle('MTF', fontsize='x-large')

    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(u, v, FF, cmap='viridis')
    ax1.set_xlabel('freq x')
    ax1.set_ylabel('freq y')
    ax1.set_zlabel('magnitude')

    return fig


def plot_MTF_logpower(FF):
    """ represent MTF using log power along spatial frequency x and spatial frequncy y=0\\
        logpower vs spatial frequency x\\

        FF  : Any   MTF FF\\
    """
    x = np.arange(-256//2, 256//2)  # remapping the x axis in range (-128, 128)
    y = 20*np.log10( FF[:,128] )    # index 128 will choose the middle data/zero frequency

    fig = plt.figure(); plt.suptitle('MTF Log Power', fontsize='x-large')

    ax1 = fig.add_subplot(111)
    ax1.plot(x, y)
    ax1.set_title('20*log10(MTF) vs freq x')
    ax1.set_xlabel('freq x')
    ax1.set_ylabel('log power')
    plt.grid()

    return fig


def analysis_single_PSF(PSF:int, isSave:bool=1, isShow:bool=0):
    """ analysis single PSF\\
        - get the PSF H\\
        - get the OTF and MTF\\
        - plot PSF 2d 3d\\
        - plot OTF complex 3d, 2d for real imaginer\\
        - plot MTF 3d\\
        - plot MTF log power\\
        
        PSF : int   the size of PSF\\
        isSave  : bool
        isShow  : bool
    """
    H, u, v = get_PSF(PSF)
    F, FF = get_OTF_MTF(H)

    fig1 = plot_PSF(u, v, H)
    if isSave: fig1.savefig('./midtest/img/no02_psf_%s.jpg' %PSF, dpi=144)

    fig2 = plot_OTF_complex(u, v, F)
    if isSave: fig2.savefig('./midtest/img/no02_otf_real_imag_%s.jpg' %PSF, dpi=144)

    fig3 = plot_MTF(u, v, FF)
    if isSave: fig3.savefig('./midtest/img/no02_mtf_%s.jpg' %PSF, dpi=144)

    fig4 = plot_MTF_logpower(FF)
    if isSave: fig4.savefig('./midtest/img/no02_mtf_logpower_%s.jpg' %PSF, dpi=144)
    
    if isShow: plt.show()


if __name__ == "__main__":
    analysis_single_PSF(PSF=30, isSave=1, isShow=0)