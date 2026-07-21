#!/usr/bin/env python3
"""One-off: load the aggregate free-behavior (GG) pickle and resave as parquet."""
import os
import sys
import time
import traceback

import pandas as pd

BASE = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker'
PKL = os.path.join(BASE, 'transformed_data_GG.pkl')
PQ  = os.path.join(BASE, 'transformed_data_GG.parquet')


def main():
    t0 = time.time()
    print('Loading pkl: {}'.format(PKL), flush=True)
    df = pd.read_pickle(PKL)
    print('Loaded shape {} in {:.1f}s'.format(df.shape, time.time() - t0),
          flush=True)

    t1 = time.time()
    tmp = PQ + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, PQ)
    print('Saved parquet: {} in {:.1f}s'.format(PQ, time.time() - t1),
          flush=True)
    print('Parquet size (GB): {:.2f}'.format(os.path.getsize(PQ) / 1e9),
          flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
