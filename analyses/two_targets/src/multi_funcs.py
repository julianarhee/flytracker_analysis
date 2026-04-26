import libs.utils as util

# Transform data
def add_pairwise_metrics(trk_, feat_, calib, flyid1=0, flyid2=1):
    ppm = calib.get('PPM', 1)
    for fid, oid in [(flyid1, flyid2), (flyid2, flyid1)]:
        f_trk = trk_[trk_['id']==fid]
        o_trk = trk_[trk_['id']==oid]
        dist = util.compute_dist_to_other(
            f_trk['pos_x'].values, f_trk['pos_y'].values,
            o_trk['pos_x'].values, o_trk['pos_y'].values,
            pix_per_mm=ppm)
        feat_.loc[feat_['id']==fid, 'dist_to_other'] = dist
        facing_angle = util.compute_facing_angle(
            f_trk['ori'].values, f_trk['pos_x'].values, f_trk['pos_y'].values,
            o_trk['pos_x'].values, o_trk['pos_y'].values)
        feat_.loc[feat_['id']==fid, 'facing_angle'] = facing_angle

    return trk_, feat_

