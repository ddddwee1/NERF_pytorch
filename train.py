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
    net_fine = model.get_nerf_model(config.net_depth_fine, config.net_width_fine, skips, input_chn, input_chn_views)
    return embed_fn, embeddirs_fn, net, net_fine

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
    return rgb_map, weights

def sample_cdf(z_vals, weights, det=False):
    bins = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
    weights = weights + 1e-5 
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

    if det:
        u = torch.linspace(0., 1., config.N_samples)
        u = torch.broadcast_to(u, list(cdf.shape[:-1])+[config.N_samples])
    else:
        u = torch.rand(cdf.shape[0], config.N_samples)
    u = u.cuda(cdf.device)
    
    idxs = torch.searchsorted(cdf, u, right=True)
    below = torch.maximum(torch.zeros_like(idxs), idxs-1).long()
    above = torch.minimum(torch.ones_like(idxs)*(cdf.shape[-1]-1), idxs).long()

    # idxs_g = torch.stack([below, above], -1)
    cdf_below = torch.gather(cdf, dim=1, index=below)
    cdf_above = torch.gather(cdf, dim=1, index=above)

    bin_below = torch.gather(bins, dim=1, index=below)
    bin_above = torch.gather(bins, dim=1, index=above)

    denom = cdf_above - cdf_below
    denom = torch.clamp(denom, 1e-5, 99999)
    t = (u - cdf_below) / denom
    samples = bin_below + t * (bin_above - bin_below)
    return samples 

def render_rays(rays_o, rays_d, viewdirs_in, near, far, embed_fn, embeddirs_fn, net, net_fine):
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
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., : , None]
    viewdirs = viewdirs_in[:,None]
    viewdirs = viewdirs.repeat(1, pts.shape[1], 1)
    # reshape them 
    shape_pts = pts.shape
    shape_dir = viewdirs.shape  
    pts = pts.reshape(-1, shape_pts[-1])
    viewdirs = viewdirs.reshape(-1, shape_dir[-1])
    embed_pts = embed_fn(pts)
    embed_dir = embeddirs_fn(viewdirs)
    raw = net(embed_pts.detach(), embed_dir.detach())
    raw = raw.reshape(shape_pts[0], shape_pts[1], -1)
    rgb_map0, weights0 = raw2outputs(raw, z_vals, rays_d)

    # fine net
    cdf_samples = sample_cdf(z_vals, weights0[...,1:-1].detach(), False) # from weight, get the cdf and do re-sampling 
    z_vals, _ = torch.sort(torch.cat([z_vals, cdf_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., : , None]
    viewdirs = viewdirs_in[:,None]
    viewdirs = viewdirs.repeat(1, pts.shape[1], 1)
    # reshape them 
    shape_pts = pts.shape
    shape_dir = viewdirs.shape  
    pts = pts.reshape(-1, shape_pts[-1])
    viewdirs = viewdirs.reshape(-1, shape_dir[-1])
    embed_pts = embed_fn(pts)
    embed_dir = embeddirs_fn(viewdirs)
    raw = net_fine(embed_pts.detach(), embed_dir.detach())
    raw = raw.reshape(shape_pts[0], shape_pts[1], -1)
    # raw = torch.ones_like(raw).cuda(raw.device)
    rgb_map, weights = raw2outputs(raw, z_vals, rays_d)
    return rgb_map0, rgb_map
    
def render(rays_o,rays_d, near, far, embed_fn, embeddirs_fn, net, net_fine):
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
    rgb_map = render_rays(rays_o, rays_d, viewdirs, near, far, embed_fn, embeddirs_fn, net, net_fine)
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
embed_fn, embeddirs_fn, net, net_fine = create_nerf()
embed_fn.cuda()
embeddirs_fn.cuda()
net.cuda()
net_fine.cuda()
saver = M.Saver(net)
saver.restore('./model/')
saver_fine = M.Saver(net_fine)
saver_fine.restore('./model_fine/')
optim = torch.optim.Adam([{'params':net.parameters()}, {'params':net_fine.parameters()}], 0.0005)

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
    rgb0, rgb = render(rays_o, rays_d, 2, 6, embed_fn, embeddirs_fn, net, net_fine)
    err0 = torch.pow(rgb0-target_s, 2).mean()
    err1 = torch.pow(rgb-target_s, 2).mean()
    err = err0 + err1 
    err.backward()
    optim.step()

    outstr = 'LsCoarse: %.4f  LsFine: %.4f'%(err0.cpu().detach().numpy(), err1.cpu().detach().numpy())
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
                    _, rgb_row = render(rays_o[row], rays_d[row], 2, 6, embed_fn, embeddirs_fn, net, net_fine)
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
        saver_fine.save('./model_fine/%08d.pth'%i)
saver.save('./model/%08d.pth'%i)
saver_fine.save('./model_fine/%08d.pth'%i)
