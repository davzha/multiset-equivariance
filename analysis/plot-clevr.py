# %%
import torch
from PIL import Image
import numpy as np
import data
from scipy.optimize import linear_sum_assignment


# %%
outputs5 = torch.load('clevr5.pth')
outputs10 = torch.load('clevr256 10.pth')
outputs20 = torch.load('clevr256 20.pth')
outputs40 = torch.load('clevr256 40.pth')


# %%
def take(iterable, n):
    l = []
    for _ in range(n):
        l.append(next(iterable))
    return l


def object_to_attributes(vec):
    tokens = iter(vec)
    coord = take(tokens, 3)
    material = np.argmax(take(tokens, 2))
    color = np.argmax(take(tokens, 8))
    shape = np.argmax(take(tokens, 3))
    size = np.argmax(take(tokens, 2))
    score = take(tokens, 1)
    access = lambda x, i: data.CLASSES[x][i]

    return (
        [3 * float(x) for x in coord],
        access("size", size),
        access("color", color),
        access("material", material),
        access("shape", shape),
        float(score[0]),
    )


def format_element(element, reference=None):
    coord_string = '({:.2f} {:.2f} {:.2f})'.format(*element[0])

    format_red = lambda s, cond: r'\wrong{' + s + r'}' if cond else s
    attribute_coloring = [format_red(e, e != r) for e, r in zip(element[1:-1], reference[1:-1])]
    attribute_string = ' '.join(attribute_coloring)

    if reference is not element:
        dist = torch.FloatTensor(element[0]).dist(torch.FloatTensor(reference[0]))
        coord_string += ' ' + format_red(f'd={dist:.2f}', dist > 0.25)
    
    s = f'{coord_string}\\vspace{{-0.7mm}}\\\\\\vspace{{0.25mm}}{attribute_string}'

    return s


def match(elements, reference):
    elements = torch.stack(elements)
    reference = torch.stack(reference)
    cost = torch.cdist(reference.unsqueeze(0), elements.unsqueeze(0)).squeeze(0)
    indices = linear_sum_assignment(cost)
    elements = elements.index_select(0, torch.LongTensor(indices[1]))
    return elements


indices = [80, 67, 76, 51, 9]

print(r'''
\setlength{\tabcolsep}{2pt}
\centering
\begin{tabular}{ccccc}
    \toprule
    Image & Ground-truth & iDSPN 10 iterations & iDSPN 20 iterations & iDSPN 40 iterations\\
''')

batch = 0
all_outputs = list(zip(*outputs5[batch], *outputs10[batch], *outputs20[batch], *outputs40[batch]))
for idx in indices:
# for i, (output5, gt_output, _, output10, _, _, output20, _, _, output40, _, _) in :
    output5, gt_output, _, output10, _, _, output20, _, _, output40, _, _ = all_outputs[idx]
    # if i not in idx:
    #     continue

    # filter out low confidence elements
    gt_output, output5, output10, output20, output40 = [list(filter(lambda x: x[-1] > 0.5, elements)) for elements in [gt_output, output5, output10, output20, output40]]
    # sort gt by x coordinate
    gt_output = sorted(gt_output, key=lambda x: x.tolist())
    # reorder predictions to correspond to gt
    output5, output10, output20, output40 = [match(elements, gt_output) for elements in [output5, output10, output20, output40]]
    # turn vectors into coords and strings
    gt_output, output5, output10, output20, output40 = [[object_to_attributes(x) for x in elements] for elements in [gt_output, output5, output10, output20, output40]]
    # format the attributes
    runs = [r'\\'.join(format_element(element, reference) for element, reference in zip(elements, gt_output)) for elements in [gt_output, output10, output20, output40]]
    print(
r'''
    \midrule
    \makecell{\includegraphics[width=0.25\textwidth, height=0.25\textwidth]{''' + f'figures/256 CLEVR_val_{idx:06d}.png' + r'''}} &
''' + ' & '.join(map('\\makecell{{ {} }}'.format, runs)) + r'''\\''')

print(r'''
    \bottomrule
\end{tabular}
''')

# %%

clevr_path = '../idspn-paper/figures/CLEVR_v1.0/images/val'
img_size = 128
for idx in indices:
    filename = f'CLEVR_val_{idx:06d}.png'
    img = Image.open(f'{clevr_path}/{filename}')
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    img.save(f'{img_size} {filename}')
