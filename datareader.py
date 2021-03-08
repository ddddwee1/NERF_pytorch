import os 
import numpy as np 
import imageio 
import json
import cv2 
from tqdm import tqdm 

trans_t = lambda t : np.float32([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
])

rot_phi = lambda phi : np.float32([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
])

rot_theta = lambda th : np.float32([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
])

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def load_blender_data(basedir, half_res=False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_%s.json'%s)) as fp:
            metas[s] = json.load(fp)
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        for frame in tqdm(meta['frames']):
            fname = os.path.join(basedir, frame['file_path']) + '.png'
            im = imageio.imread(fname)
            if half_res:
                im = cv2.resize(im, (400,400))
            imgs.append(im)
            poses.append(np.float32(frame['transform_matrix']))
        imgs = np.float32(imgs) / 255.
        poses = np.float32(poses)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H,W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    if half_res:
        H = H//2
        W = W//2
        focal = focal / 2. 
    return imgs, poses, render_poses, [H,W,focal], i_split

if __name__=='__main__':
    imgs, poses, render_poses, hwf, i_split = load_blender_data('./data/nerf_synthetic/lego')
    print(imgs[0].max(), imgs[0].min(), poses[0])
