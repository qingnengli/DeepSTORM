
import numpy as np
from scipy import io as sio

def list2array(image,batch_size):
    array=[]
    for i in range(batch_size):
        xy = np.squeeze(image[i])
        array.append(xy)
    array=np.expand_dims(array,3)
    return array

def loadmat(issim=True,RAW=False):
    if RAW:
        IM = 'RAW'
    else:
        IM ='TIF'
    if issim:
        path = '/home/amax/SIAT/DeepSTORM/data'
        L1 = sio.loadmat(path + '/0.1density_px100_IG_no30_tif.mat')[IM]
        L2 = sio.loadmat(path + '/0.1density_px100_G_no30_tif.mat')[IM]
        L3 = sio.loadmat(path + '/0.2density_px80_IG_no30_tif.mat')[IM]
        L4 = sio.loadmat(path + '/0.2density_px80_G_no30_tif.mat')[IM]
        L5 = sio.loadmat(path + '/0.3density_px150_IG_no30_tif.mat')[IM]
        L6 = sio.loadmat(path + '/0.3density_px150_G_no30_tif.mat')[IM]
        L7 = sio.loadmat(path + '/0.4density_px100_IG_no60_tif.mat')[IM]
        L8 = sio.loadmat(path + '/0.4density_px100_G_no60_tif.mat')[IM]
        L9 = sio.loadmat(path + '/0.5density_px80_IG_no60_tif.mat')[IM]
        L10 = sio.loadmat(path + '/0.5density_px80_G_no60_tif.mat')[IM]
        L11 = sio.loadmat(path + '/0.6density_px150_IG_no60_tif.mat')[IM]
        L12 = sio.loadmat(path + '/0.6density_px150_G_no60_tif.mat')[IM]
        L13 = sio.loadmat(path + '/0.7density_px100_IG_no90_tif.mat')[IM]
        L14 = sio.loadmat(path + '/0.7density_px100_G_no90_tif.mat')[IM]
        L15 = sio.loadmat(path + '/0.8density_px80_IG_no90_tif.mat')[IM]
        L16 = sio.loadmat(path + '/0.8density_px80_G_no90_tif.mat')[IM]
        L17 = sio.loadmat(path + '/0.9density_px150_IG_no90_tif.mat')[IM]
        L18 = sio.loadmat(path + '/0.9density_px150_G_no90_tif.mat')[IM]
        L19 = sio.loadmat(path + '/1density_px120_IG_no120_tif.mat')[IM]
        L20 = sio.loadmat(path + '/1density_px120_G_no120_tif.mat')[IM]

        L1_xy= sio.loadmat(path + '/0.1density_px100_IG_no30_cord.mat')['cord']
        L2_xy= sio.loadmat(path + '/0.1density_px100_G_no30_cord.mat')['cord']
        L3_xy= sio.loadmat(path + '/0.2density_px80_IG_no30_cord.mat')['cord']
        L4_xy= sio.loadmat(path + '/0.2density_px80_G_no30_cord.mat')['cord']
        L5_xy= sio.loadmat(path + '/0.3density_px150_IG_no30_cord.mat')['cord']
        L6_xy= sio.loadmat(path + '/0.3density_px150_G_no30_cord.mat')['cord']
        L7_xy= sio.loadmat(path + '/0.4density_px100_IG_no60_cord.mat')['cord']
        L8_xy= sio.loadmat(path + '/0.4density_px100_G_no60_cord.mat')['cord']
        L9_xy= sio.loadmat(path + '/0.5density_px80_IG_no60_cord.mat')['cord']
        L10_xy= sio.loadmat(path + '/0.5density_px80_G_no60_cord.mat')['cord']
        L11_xy= sio.loadmat(path + '/0.6density_px150_IG_no60_cord.mat')['cord']
        L12_xy= sio.loadmat(path + '/0.6density_px150_G_no60_cord.mat')['cord']
        L13_xy= sio.loadmat(path + '/0.7density_px100_IG_no90_cord.mat')['cord']
        L14_xy= sio.loadmat(path + '/0.7density_px100_G_no90_cord.mat')['cord']
        L15_xy= sio.loadmat(path + '/0.8density_px80_IG_no90_cord.mat')['cord']
        L16_xy= sio.loadmat(path + '/0.8density_px80_G_no90_cord.mat')['cord']
        L17_xy= sio.loadmat(path + '/0.9density_px150_IG_no90_cord.mat')['cord']
        L18_xy= sio.loadmat(path + '/0.9density_px150_G_no90_cord.mat')['cord']
        L19_xy= sio.loadmat(path + '/1density_px120_IG_no120_cord.mat')['cord']
        L20_xy= sio.loadmat(path + '/1density_px120_G_no120_cord.mat')['cord']

        H1 = sio.loadmat(path + '/2density_px100_IG_no30_tif.mat')[IM]
        H2 = sio.loadmat(path + '/2density_px100_G_no30_tif.mat')[IM]
        H3 = sio.loadmat(path + '/2.5density_px80_IG_no30_tif.mat')[IM]
        H4 = sio.loadmat(path + '/2.5density_px80_G_no30_tif.mat')[IM]
        H5 = sio.loadmat(path + '/3density_px150_IG_no30_tif.mat')[IM]
        H6 = sio.loadmat(path + '/3density_px150_G_no30_tif.mat')[IM]
        H7 = sio.loadmat(path + '/3.5density_px100_IG_no60_tif.mat')[IM]
        H8 = sio.loadmat(path + '/3.5density_px100_G_no60_tif.mat')[IM]
        H9 = sio.loadmat(path + '/4density_px80_IG_no60_tif.mat')[IM]
        H10 = sio.loadmat(path + '/4density_px80_G_no60_tif.mat')[IM]
        H11 = sio.loadmat(path + '/4.5density_px150_IG_no60_tif.mat')[IM]
        H12 = sio.loadmat(path + '/4.5density_px150_G_no60_tif.mat')[IM]
        H13 = sio.loadmat(path + '/5density_px100_IG_no90_tif.mat')[IM]
        H14 = sio.loadmat(path + '/5density_px100_G_no90_tif.mat')[IM]
        H15 = sio.loadmat(path + '/5.5density_px80_IG_no90_tif.mat')[IM]
        H16 = sio.loadmat(path + '/5.5density_px80_G_no90_tif.mat')[IM]
        H17 = sio.loadmat(path + '/6density_px150_IG_no90_tif.mat')[IM]
        H18 = sio.loadmat(path + '/6density_px150_G_no90_tif.mat')[IM]
        H19 = sio.loadmat(path + '/6.5density_px120_IG_no120_tif.mat')[IM]
        H20 = sio.loadmat(path + '/6.5density_px120_G_no120_tif.mat')[IM]

        H1_xy = sio.loadmat(path + '/2density_px100_IG_no30_cord.mat')['cord']
        H2_xy = sio.loadmat(path + '/2density_px100_G_no30_cord.mat')['cord']
        H3_xy = sio.loadmat(path + '/2.5density_px80_IG_no30_cord.mat')['cord']
        H4_xy = sio.loadmat(path + '/2.5density_px80_G_no30_cord.mat')['cord']
        H5_xy = sio.loadmat(path + '/3density_px150_IG_no30_cord.mat')['cord']
        H6_xy = sio.loadmat(path + '/3density_px150_G_no30_cord.mat')['cord']
        H7_xy = sio.loadmat(path + '/3.5density_px100_IG_no60_cord.mat')['cord']
        H8_xy = sio.loadmat(path + '/3.5density_px100_G_no60_cord.mat')['cord']
        H9_xy = sio.loadmat(path + '/4density_px80_IG_no60_cord.mat')['cord']
        H10_xy = sio.loadmat(path + '/4density_px80_G_no60_cord.mat')['cord']
        H11_xy = sio.loadmat(path + '/4.5density_px150_IG_no60_cord.mat')['cord']
        H12_xy = sio.loadmat(path + '/4.5density_px150_G_no60_cord.mat')['cord']
        H13_xy = sio.loadmat(path + '/5density_px100_IG_no90_cord.mat')['cord']
        H14_xy = sio.loadmat(path + '/5density_px100_G_no90_cord.mat')['cord']
        H15_xy = sio.loadmat(path + '/5.5density_px80_IG_no90_cord.mat')['cord']
        H16_xy = sio.loadmat(path + '/5.5density_px80_G_no90_cord.mat')['cord']
        H17_xy = sio.loadmat(path + '/6density_px150_IG_no90_cord.mat')['cord']
        H18_xy = sio.loadmat(path + '/6density_px150_G_no90_cord.mat')['cord']
        H19_xy = sio.loadmat(path + '/6.5density_px120_IG_no120_cord.mat')['cord']
        H20_xy = sio.loadmat(path + '/6.5density_px120_G_no120_cord.mat')['cord']
        LS = np.concatenate((L1, L2, L3, L4, L5, L6, L7, L8, L9, L10,L11,L12,L13,L14,L15,L16,L17,L18,L19,L20), 1)[0]
        LS_xy = np.concatenate((L1_xy, L2_xy, L3_xy, L4_xy, L5_xy, L6_xy, L7_xy, L8_xy, L9_xy, L10_xy,
                                L11_xy, L12_xy, L13_xy, L14_xy, L15_xy, L16_xy, L17_xy, L18_xy, L19_xy, L20_xy), 1)[0]
        HD = np.concatenate((H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16, H17,H18,H19,H20), 1)[0]
        HD_xy = np.concatenate((H1_xy, H2_xy, H3_xy, H4_xy, H5_xy, H6_xy, H7_xy, H8_xy, H9_xy, H10_xy,
                                H11_xy, H12_xy, H13_xy, H14_xy, H15_xy, H16_xy, H17_xy,H18_xy,H19_xy,H20_xy), 1)[0]



        # print(LS.shape, LS_xy.shape)
        # print(HD.shape, HD_xy.shape)
        train_images = np.concatenate((LS, HD), 0)
        train_images = list2array(train_images,len(train_images))
        train_cords = np.concatenate((LS_xy, HD_xy), 0)
        return train_images, train_cords

    else:
        path = '/home/amax/SIAT/Data-SMLM/train/'
        BT_HD = np.load(path + 'IMG/NPY/Bundled_Tubes_HD' + '_' + IM +'.npy')
        BT_LS = np.load(path + 'IMG/NPY/Bundled_Tubes_LS' + '_' + IM + '.npy')
        MT0_N1_HD = np.load(path + 'IMG/NPY/MT0_N1_HD' + '_' + IM + '.npy')
        MT0_N1_LS = np.load(path + 'IMG/NPY/MT0_N1_LS' + '_' + IM + '.npy')
        MT0_N2_HD = np.load(path + 'IMG/NPY/MT0_N2_HD' + '_' + IM + '.npy')
        MT0_N2_LS = np.load(path + 'IMG/NPY/MT0_N2_LS' + '_' + IM + '.npy')

        BT_HD_xy = sio.loadmat(path+'SR/Bundled_Tubes_HD' + '_cord.mat')['cord']
        BT_LS_xy = sio.loadmat(path+'SR/Bundled_Tubes_LS' + '_cord.mat')['cord']
        MT0_N1_HD_xy = sio.loadmat(path + 'SR/MT0_N1_HD' + '_cord.mat')['cord']
        MT0_N1_LS_xy = sio.loadmat(path + 'SR/MT0_N1_LS' + '_cord.mat')['cord']
        MT0_N2_HD_xy = sio.loadmat(path + 'SR/MT0_N2_HD' + '_cord.mat')['cord']
        MT0_N2_LS_xy = sio.loadmat(path + 'SR/MT0_N2_LS' + '_cord.mat')['cord']

        image = np.concatenate((BT_HD,BT_LS,MT0_N1_HD,MT0_N1_LS,MT0_N2_HD,MT0_N2_LS),0)
        cord = np.concatenate((BT_HD_xy,BT_LS_xy,MT0_N1_HD_xy,MT0_N1_LS_xy,MT0_N2_HD_xy,MT0_N2_LS_xy),1)[0]
        return image,cord
