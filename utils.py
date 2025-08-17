import xarray as xr
from climpred import HindcastEnsemble


def calc_forecast_skill(fcst_ds, ref_ds, metric='pearson_r', is_mv3=True, comparison="e2r", 
                        by_month=False, verify_periods=slice('1979-01', '2022-12')):
    try:
        fcst_ds = fcst_ds.squeeze().drop('member')
    except:
        pass

    if is_mv3:
        fcst_ds = fcst_ds.rolling(init=3, center=True, min_periods=1).mean('init')
        ref_mv3 = ref_ds.rolling(time=3, center=True, min_periods=1).mean().dropna(dim='time')
    else:
        ref_mv3 = ref_ds

    hc_XRO = HindcastEnsemble(fcst_ds.sel(init=verify_periods))
    hc_XRO.add_reference(ref_mv3, 'observations')
    if by_month:
        # Note: groupby functionality may need to be handled differently in newer climpred
        skill_XRO = hc_XRO.compute_metric(refname='observations', metric=metric, comparison=comparison)
    else:
        skill_XRO = hc_XRO.compute_metric(refname='observations', metric=metric, comparison=comparison)
    try:
        del skill_XRO.attrs['skipna']
        skill_XRO = skill_XRO.drop('skill')
    except:
        pass

    for var in skill_XRO.data_vars:
        if var != 'model':
            skill_XRO[var].encoding['dtype'] = 'float32'
            skill_XRO[var].encoding['_FillValue'] = 1e20
    return skill_XRO


