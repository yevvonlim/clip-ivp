import jsonlines
import matplotlib.pyplot as plt
import numpy as np

str_plt_style = 'bmh'
plt.style.use([str_plt_style])
fig, ax = plt.subplots(1,1,figsize=(13,5))
plt.ylabel('FID')
plt.xlabel('Iterations')

root_dir = '/root/stylegan2-ada-intraoral/class-conditional/'
keys = ['gamma5', 'gamma10', 'gamma15']
progress_path = {key:f'{root_dir}cond-auto1-{key}-resumecustom/metric-fid50k_full.jsonl' for key in keys if key != 'gamma10'}
progress_path['gamma10'] = '/root/stylegan2-ada-intraoral/class-conditional/gamma10/metric-fid50k_full.jsonl'
progress = {}
for key in keys:
    fid = []
    steps = []
    with jsonlines.open(progress_path[key]) as f:
        for line in f:
            if int(line['snapshot_pkl'].replace('network-snapshot-', '').replace('.pkl', '')) > 15000:
                break
            fid.append(line['results']['fid50k_full'])
            steps.append(int(line['snapshot_pkl'].replace('network-snapshot-', '').replace('.pkl', '')))
    progress[key] = list(zip(steps, fid))
    minidx = np.argmin(fid)
    plt.plot(steps, fid, label=key)
    plt.scatter(steps[minidx], fid[minidx], color='black')

# Function x**(1/2)
def forward(x):
    return np.power(x, 1/2) 


def inverse(x):
    return np.power(x, 2) 

# plt.xticks([0, 1000, 2000, 4000, 8000, 15000])
steps_ = [0,  1000, 2000, 4000, 8000, 12000, 15000]
labels_ = [f'{int(step/1000)}M' if step != 0 and step%1000==0 else f'{step/1000}k' if step%1000 != 0 else '0' for step in steps_]

# plt.xscale('function', functions=(forward, inverse))
plt.xscale('linear')
plt.yscale('linear')
plt.yticks([3, 4, 5, 6, 15, 30])
plt.xticks(steps_, labels=labels_)
plt.ylim(top=20, bottom=1)
plt.xlim(left=0)
# plt.yscale('function', functions=(forward, inverse))
ax.legend()
plt.savefig('/root/stylegan2-ada-intraoral/gamma_sweep.png')
