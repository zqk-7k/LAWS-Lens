#!/usr/bin/env python3
"""
Faithful observable-level catalog simulator for the LensGraph experiments.

Reproduces EXACTLY the lensing physics in the repository's data-generation
scripts (data_generation/generated_10000_scripts/{SIS,PM}_GW_events_ET3.py):

  Source priors:
    mass_1_source, mass_2_source ~ Uniform(10, 70)   [M_sun]
    luminosity_distance ~ UniformComovingVolume(z in [0.01, 2])
    ra ~ Uniform(0, 2pi);  dec ~ Cosine
    geocent_time ~ Uniform(2015-09-14 .. 2025-12-10)   [~10 yr]

  SIS lens:
    z_l = z_s/2;  sigma_v ~ U(100,500) km/s
    theta_E from SIS;  y ~ U(0.01,0.3)
    mu_plus=1+1/y, mu_minus=1-1/y
    t_d = 8 * 4 pi^2 sigma_v^4 (1+z_l) Dls Dl y / (Ds c^5)   [seconds]

  PM lens:
    z_l = z_s/2;  m_l ~ U(1e8,1e10) M_sun
    y ~ U(0.01,0.3)
    mu_plus = 1/2 + (y^2+2)/(2 y sqrt(y^2+4));  mu_minus = 1/2 - ...
    t_d = 4 G m_l (1+z_l) c^-3 [ 0.5 y sqrt(y^2+4) + ln((sqrt(y^2+4)+y)/(sqrt(y^2+4)-y)) ]

The ONLY quantity not reproduced bit-exactly is the matched-filter SNR (which
needs full waveform generation on GPU). We model it physically as
    rho ∝ sqrt(|mu|) * (M_c^{5/6}) / d_L_eff
which is the correct leading-order scaling, then calibrate the constant so the
network-SNR distribution matches the repo's reported ET3 / LIGO ranges.

A90 (90% sky area) is modelled SNR-dependently exactly as in the repo's
observed-sky scheme: A90 = A90_ref (rho_ref/rho_net)^2 * lognormal, clipped.
"""
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as const
from astropy.time import Time

# ---- physical constants (SI) ----
G = const.G.value
c = const.c.value
Msun = const.M_sun.value
Mpc_to_m = (1.0 * u.Mpc).to(u.m).value
DEG2RAD = np.pi / 180.0

T_START = Time('2015-09-14 09:50:45.39', scale='utc').gps
T_END   = Time('2025-12-10 17:18:45.39', scale='utc').gps
T_SPAN_YR = (T_END - T_START) / (365.25 * 24 * 3600)


def _sample_redshift_comoving(n, rng, z_min=0.01, z_max=2.0, ngrid=2000):
    """Sample z ~ uniform in comoving volume (matches UniformComovingVolume)."""
    zg = np.linspace(z_min, z_max, ngrid)
    dVdz = cosmo.differential_comoving_volume(zg).value  # Mpc^3/sr
    cdf = np.cumsum(dVdz); cdf /= cdf[-1]
    u_ = rng.uniform(0, 1, n)
    return np.interp(u_, cdf, zg)


def _angular_diameter_distances(z_l, z_s):
    """Return Dl, Ds, Dls in metres (angular diameter distances)."""
    Dl = cosmo.angular_diameter_distance(z_l).value * Mpc_to_m
    Ds = cosmo.angular_diameter_distance(z_s).value * Mpc_to_m
    Dls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value * Mpc_to_m
    return Dl, Ds, Dls


def _chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def simulate_catalog(n_systems_sis, n_systems_pm, n_unlensed,
                     detector='ET3', seed=0, snr_threshold=8.0,
                     time_span_yr=None):
    """
    Build an observable-level catalog with faithful lensing physics.

    time_span_yr: if given, OVERRIDES the ~10yr window (used for the
    event-density stress test). Smaller span at fixed N => higher density.

    Returns a DataFrame with one row per detected image/event:
      event_id, source_id, kind ('SIS','PM','unlensed'),
      image (0/1 for lensed, -1 for unlensed),
      geocent_time, ra, dec, mass_1, mass_2, chirp_mass, mass_ratio,
      luminosity_distance, z_s, mu, network_snr, sky_area_90_deg2
    """
    rng = np.random.default_rng(seed)
    if time_span_yr is None:
        t_lo, t_hi = T_START, T_END
    else:
        # centre a window of the requested span on the original window centre
        mid = 0.5 * (T_START + T_END)
        half = 0.5 * time_span_yr * 365.25 * 24 * 3600
        t_lo, t_hi = mid - half, mid + half

    # detector-dependent SNR calibration + A90 reference
    # (calibrated so network-SNR medians match repo: ET3 strong, LIGO weaker)
    if detector == 'ET3':
        snr_norm = 4200.0   # calibrated: ET3 A90 median ~20 deg^2
        a90_ref, rho_ref, a90_lo, a90_hi = 20.0, 30.0, 5.0, 200.0
    else:  # LIGO H1L1
        snr_norm = 1100.0
        a90_ref, rho_ref, a90_lo, a90_hi = 400.0, 20.0, 50.0, 1000.0

    rows = []
    sid = 0

    def _source_block(n):
        m1 = rng.uniform(10, 70, n)
        m2 = rng.uniform(10, 70, n)
        # ensure m1>=m2
        m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
        z_s = _sample_redshift_comoving(n, rng)
        dL = cosmo.luminosity_distance(z_s).value  # Mpc
        ra = rng.uniform(0, 2 * np.pi, n)
        dec = np.arcsin(rng.uniform(-1, 1, n))  # Cosine prior on dec
        t = rng.uniform(t_lo, t_hi, n)
        return m1, m2, z_s, dL, ra, dec, t

    def _snr(mc, dL_eff, mu):
        # leading-order: rho ∝ |mu|^{1/2} Mc^{5/6} / dL
        base = snr_norm * np.sqrt(np.abs(mu)) * (mc ** (5.0 / 6.0)) / dL_eff
        # add mild scatter (detector noise realisation)
        return base * rng.lognormal(0.0, 0.10, size=np.shape(base))

    def _a90(rho):
        a = a90_ref * (rho_ref / np.maximum(rho, 1e-3)) ** 2 * rng.lognormal(0, 0.35, size=np.shape(rho))
        return np.clip(a, a90_lo, a90_hi)

    def _scatter_sky(ra_true, dec_true, a90):
        # observed position = true + Gaussian scatter consistent with A90;
        # each image scattered INDEPENDENTLY (real measurement), so two images
        # of one source differ by ~sqrt(2)*sigma in observed position.
        sigma = float(np.sqrt(a90 / np.pi) * DEG2RAD / np.sqrt(2.0 * np.log(10.0)))
        dra = rng.normal(0, sigma) / max(np.cos(dec_true), 1e-3)
        ddec = rng.normal(0, sigma)
        return (ra_true + dra) % (2 * np.pi), float(np.clip(dec_true + ddec, -np.pi/2, np.pi/2))

    # ---------- SIS lensed systems ----------
    if n_systems_sis > 0:
        m1, m2, z_s, dL, ra, dec, t1 = _source_block(n_systems_sis)
        mc = _chirp_mass(m1, m2); q = m2 / m1
        z_l = z_s / 2
        sigma_v = rng.uniform(100, 500, n_systems_sis) * 1000.0  # m/s
        y = rng.uniform(0.01, 0.3, n_systems_sis)
        mu_plus = 1 + 1.0 / y
        mu_minus = 1 - 1.0 / y
        Dl, Ds, Dls = _angular_diameter_distances(z_l, z_s)
        t_d = 8 * 4 * np.pi**2 * sigma_v**4 * (1 + z_l) * Dls * Dl * y / (Ds * c**5)
        for i in range(n_systems_sis):
            # image 1 (mu_plus), image 2 (mu_minus), delayed by t_d
            for img, (mu, tt) in enumerate([(mu_plus[i], t1[i]),
                                            (mu_minus[i], t1[i] + t_d[i])]):
                rho = _snr(mc[i], dL[i], mu)
                if rho < snr_threshold:
                    continue
                a90_img = float(_a90(rho))
                ra_obs, dec_obs = _scatter_sky(ra[i], dec[i], a90_img)
                rows.append(dict(source_id=sid, kind='SIS', image=img,
                                 geocent_time=tt, ra=ra_obs, dec=dec_obs,
                                 mass_1=m1[i], mass_2=m2[i], chirp_mass=mc[i],
                                 mass_ratio=q[i], luminosity_distance=dL[i],
                                 z_s=z_s[i], mu=mu, network_snr=rho,
                                 sky_area_90_deg2=a90_img))
            sid += 1

    # ---------- PM lensed systems ----------
    if n_systems_pm > 0:
        m1, m2, z_s, dL, ra, dec, t1 = _source_block(n_systems_pm)
        mc = _chirp_mass(m1, m2); q = m2 / m1
        z_l = z_s / 2
        m_l = rng.uniform(1e8, 1e10, n_systems_pm) * Msun
        y = rng.uniform(0.01, 0.3, n_systems_pm)
        s = np.sqrt(y**2 + 4)
        mu_plus = 0.5 + (y**2 + 2) / (2 * y * s)
        mu_minus = 0.5 - (y**2 + 2) / (2 * y * s)
        t_d = 4 * G * m_l * (1 + z_l) * c**(-3) * (0.5 * y * s + np.log((s + y) / (s - y)))
        for i in range(n_systems_pm):
            for img, (mu, tt) in enumerate([(mu_plus[i], t1[i]),
                                            (mu_minus[i], t1[i] + t_d[i])]):
                rho = _snr(mc[i], dL[i], mu)
                if rho < snr_threshold:
                    continue
                a90_img = float(_a90(rho))
                ra_obs, dec_obs = _scatter_sky(ra[i], dec[i], a90_img)
                rows.append(dict(source_id=sid, kind='PM', image=img,
                                 geocent_time=tt, ra=ra_obs, dec=dec_obs,
                                 mass_1=m1[i], mass_2=m2[i], chirp_mass=mc[i],
                                 mass_ratio=q[i], luminosity_distance=dL[i],
                                 z_s=z_s[i], mu=mu, network_snr=rho,
                                 sky_area_90_deg2=a90_img))
            sid += 1

    # ---------- unlensed background ----------
    if n_unlensed > 0:
        m1, m2, z_s, dL, ra, dec, t = _source_block(n_unlensed)
        mc = _chirp_mass(m1, m2); q = m2 / m1
        for i in range(n_unlensed):
            rho = _snr(mc[i], dL[i], 1.0)
            if rho < snr_threshold:
                continue
            a90_img = float(_a90(rho))
            ra_obs, dec_obs = _scatter_sky(ra[i], dec[i], a90_img)
            rows.append(dict(source_id=sid, kind='unlensed', image=-1,
                             geocent_time=t[i], ra=ra_obs, dec=dec_obs,
                             mass_1=m1[i], mass_2=m2[i], chirp_mass=mc[i],
                             mass_ratio=q[i], luminosity_distance=dL[i],
                             z_s=z_s[i], mu=1.0, network_snr=rho,
                             sky_area_90_deg2=a90_img))
            sid += 1

    df = pd.DataFrame(rows)
    df.insert(0, 'event_id', np.arange(len(df)))
    return df


if __name__ == '__main__':
    # quick sanity: reproduce the main 9000-event ET3 scale
    df = simulate_catalog(3000, 3000, 3000, detector='ET3', seed=0)
    print("ET3 catalog:", len(df), "events")
    print(df.groupby('kind').size())
    print("\nSNR network median by kind:")
    print(df.groupby('kind')['network_snr'].median())
    print("\nA90 median by kind (deg^2):")
    print(df.groupby('kind')['sky_area_90_deg2'].median())
    # time delays
    for fam in ['SIS', 'PM']:
        sub = df[df.kind == fam]
        dts = []
        for s, g in sub.groupby('source_id'):
            if len(g) == 2:
                dts.append(abs(g.geocent_time.values[0] - g.geocent_time.values[1]))
        dts = np.array(dts)
        if len(dts):
            print(f"\n{fam} time delay: median={np.median(dts):.3g}s = {np.median(dts)/86400:.3g}days, "
                  f"p90={np.percentile(dts,90):.3g}s")
    df.to_csv('/tmp/paper/exp/catalog_et3_main.csv', index=False)
    print("\nsaved.")
