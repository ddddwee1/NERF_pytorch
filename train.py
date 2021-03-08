import numpy as np 
import torch 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import datareader
import config 
import model 
import random 
from tqdm import tqdm 
import pickle 
import os 
import imageio

def create_nerf():
    embed_fn, input_chn = model.get_embedder(config.multires)
    embeddirs_fn, input_chn_views = model.get_embedder(config.multires_views)
    skips = [4]
    net = model.get_nerf_model(config.net_depth, config.net_width, skips, input_chn, input_chn_views)
    # net_fine = model.get_nerf_model(config.net_depth_fine, config.net_width_fine, skips, input_chn, input_chn_views)
    return embed_fn, embeddirs_fn, net 

def get_rays(H, W, focal, c2w):
    i,j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def raw2alpha(raw, dists):
    return 1.0 - torch.exp(- F.relu(raw) * dists)

def raw2outputs(raw, z_vals, rays_d):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.ones_like(dists[...,:1])*1e10], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    rgb = torch.sigmoid(raw[...,:3])
    alpha = raw2alpha(raw[...,3], dists)
    # exclusive cumprod 
    weights = 1. - alpha + 1e-10
    weights = torch.cat([torch.ones_like(weights[...,:1]), weights[...,:-1]], dim=-1)
    weights = alpha * torch.cumprod( weights, dim=-1)
    rgb_map = torch.sum(weights[...,None] * rgb, dim=1)
    return rgb_map

def render_rays(rays_o, rays_d, viewdirs, near, far, embed_fn, embeddirs_fn, net):
    # N_rays = rays_o.shape[0]
    t_vals = torch.linspace(0., 1., config.N_samples)
    z_vals = near * (1. - t_vals) + far * t_vals
    # print(z_vals.shape)
    # perturb random sample point between lower and upper 
    mids = 0.5 * (z_vals[:, 1:] + z_vals[:,:-1])
    upper = torch.cat([mids, z_vals[:,-1:]], -1)
    lower = torch.cat([z_vals[:,:1], mids], -1)
    rand = torch.rand(z_vals.shape) * (upper - lower)
    z_vals = lower + rand
    z_vals = z_vals.cuda(rays_o.device)
    # get raw from embedder and network 
    # print(rays_o.shape, rays_d.shape, z_vals.shape)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., : , None]
    viewdirs = viewdirs[:,None]
    viewdirs = viewdirs.repeat(1, pts.shape[1], 1)
    # reshape them 
    shape_pts = pts.shape
    shape_dir = viewdirs.shape  
    pts = pts.reshape(-1, shape_pts[-1])
    viewdirs = viewdirs.reshape(-1, shape_dir[-1])
    embed_pts = embed_fn(pts)
    embed_dir = embeddirs_fn(viewdirs)
    raw = net(embed_pts, embed_dir)
    raw = raw.reshape(shape_pts[0], shape_pts[1], -1)
    
    rgb_map = raw2outputs(raw, z_vals, rays_d)
    return rgb_map
    
def render(rays_o,rays_d, near, far, embed_fn, embeddirs_fn, net):
    shape = rays_d.shape 
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    # reshape 
    viewdirs = viewdirs.reshape(-1, 3)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    near = near * torch.ones(rays_o.shape[0], 1)
    far = far * torch.ones(rays_o.shape[0], 1)
    # all_ret = render_rays()
    rgb_map = render_rays(rays_o, rays_d, viewdirs, near, far, embed_fn, embeddirs_fn, net)
    return rgb_map

DATAPATH = './data/nerf_synthetic/lego'

if os.path.exists('data.pkl'):
    imgs, poses, render_poses, hwf, i_split = pickle.load(open('data.pkl','rb'))
else:
    imgs, poses, render_poses, hwf, i_split = datareader.load_blender_data(DATAPATH)
    pickle.dump([imgs, poses, render_poses, hwf, i_split], open('data.pkl','wb'))
H, W, focal = hwf 

i_train, i_val, i_test = i_split

# white background 
imgs = imgs[...,:3] * imgs[...,-1:] + (1. - imgs[...,-1:])

# pre-computed rays 
if os.path.exists('rays.pkl'):
    rays = pickle.load(open('rays.pkl','rb'))
else:
    rays = []
    for i in tqdm(range(poses.shape[0])):
        pose = poses[i,:3, :4]
        rays_o, rays_d = get_rays(H,W,focal, pose)
        rays.append([rays_o, rays_d])
    pickle.dump(rays, open('rays.pkl','wb'))

# train loop 
embed_fn, embeddirs_fn, net = create_nerf()
embed_fn.cuda()
embeddirs_fn.cuda()
net.cuda()
saver = M.Saver(net)
saver.restore('./model/')
optim = torch.optim.Adam(net.parameters(), 0.0005)

bar = tqdm(range(config.N_iters))
for i in bar:
    img_idx = np.random.choice(i_train)
    target = imgs[img_idx]
    # pose = poses[img_idx, :3, :4]
    # rays_o, rays_d = get_rays(H, W, focal, pose) # can be moved to pre-computed
    rays_o, rays_d = rays[img_idx]
    rays_o = torch.from_numpy(rays_o).cuda()
    rays_d = torch.from_numpy(rays_d).cuda()
    target = torch.from_numpy(target).cuda()
    
    rand_idx = torch.from_numpy(np.int64(random.sample(range(H*W), k=config.N_rand)))
    rand_idx_x = torch.remainder(rand_idx, W)
    rand_idx_y = torch.div(rand_idx, W, rounding_mode='floor')
    rays_o = rays_o[rand_idx_y, rand_idx_x]
    rays_d = rays_d[rand_idx_y, rand_idx_x]
    target_s = target[rand_idx_y, rand_idx_x]

    optim.zero_grad()
    rgb = render(rays_o, rays_d, 2, 6, embed_fn, embeddirs_fn, net)
    err = torch.pow(rgb-target_s, 2).mean()
    err.backward()
    optim.step()

    outstr = 'Ls: %.4f'%(err.cpu().detach().numpy())
    bar.set_description(outstr)

    if i%config.render_test_iter==0 and i>0:
        print('Rendering test set...')
        if not os.path.exists('./generated_test/'):
            os.mkdir('./generated_test/')
        for img_idx in tqdm(i_test[:10]):
            rays_o, rays_d = rays[img_idx]
            rays_o = torch.from_numpy(rays_o).cuda()
            rays_d = torch.from_numpy(rays_d).cuda()
            # write batchify 
            with torch.no_grad():
                rgb = []
                for row in range(rays_o.shape[0]):
                    rgb_row = render(rays_o[row], rays_d[row], 2, 6, embed_fn, embeddirs_fn, net)
                    rgb.append(rgb_row)
                rgb = torch.stack(rgb, 0)
            rgb = rgb.cpu().numpy()
            rgb = rgb.reshape([H,W,3])
            rgb = np.uint8(255 * np.clip(rgb, 0., 1.))
            imageio.imwrite('./generated_test/%04d.jpg'%img_idx, rgb)
            
    if i%config.decay_iter==0 and i>0:
        lr = optim.param_groups[0]['lr']
        lr = lr * 0.1 
        print('Set LR to: %.1e'%lr)
        for param_group in optim.param_groups: 
            param_group['lr'] = lr 

    if i%config.save_interval==0 and i>0:
        saver.save('./model/%08d.pth'%i)
saver.save('./model/%08d.pth'%i)
