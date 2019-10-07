import argparse
import glob
import os
from shutil import copyfile
import sys

from pylego.misc import add_argument as arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg(parser, 'logs_path', type=str, default='../logs', help='input logs directory')
    arg(parser, 'out_path', type=str, default='out', help='output directory')
    arg(parser, 'name_pattern', type=str, default='*', help='names to match in logs directory')
    arg(parser, 'step', type=int, default=-1, help='override comparison step if greater than 0')
    arg(parser, 'min_step', type=int, default=-1, help='minimum step to dump if greater than 0')
    arg(parser, 'latest', type=bool, default=False, help='get the latest step for each directory')
    flags = parser.parse_args()

    print('* Reading logs')

    dir_pattern = '{}/{}'.format(flags.logs_path, flags.name_pattern)
    in_dirs = glob.glob(dir_pattern)
    if not in_dirs:
        raise ValueError('Pattern %s did not match any names' % dir_pattern)
    print('* Found input directories:\n-', '\n- '.join(in_dirs))
    choice = input('Write to %s (y/n)? ' % flags.out_path).strip().lower()
    if choice != 'y':
        sys.exit(0)
    print('* Creating output directory', flags.out_path)
    os.makedirs(flags.out_path)

    if flags.step > 0:
        best_index = flags.step
    else:
        best_index = 99999999
    img_list = {}
    for path in in_dirs:
        images = glob.glob(path + '/vis_*.png')
        if not images:
            continue
        images = sorted([(int(img.rsplit('.', 1)[0].rsplit('_', 1)[-1]), img) for img in images])
        if images[-1][0] < best_index:
            best_index = images[-1][0]
        key_name = path.rsplit('/', 1)[-1]
        img_list[key_name] = images

    print('\n* Comparison step detected:', best_index)

    best_image = {}
    for k, v in img_list.items():
        while v:
            step, fname = v.pop()
            if flags.latest or step <= best_index:
                if step >= flags.min_step:
                    best_image[k] = (fname, step)
                break
            elif len(v) == 0:
                print('! WARNING: all images from %s are after step' % k, best_index)

    print('* Copying images')
    for k, source in best_image.items():
        dest = flags.out_path + '/' + k + '_' + str(source[1]) + '.png'
        copyfile(source[0], dest)

    print('* All done!')
