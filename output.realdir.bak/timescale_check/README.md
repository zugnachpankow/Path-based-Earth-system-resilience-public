# Timescale conversion check (current vs fixed)

- I_CAL = 2.6267; fix: T_ODE = τ / I_CAL (cusp ODE time constant).
- Crossing x=-1→+1 at 4.0/1.8: current = 2.627·τ (too slow), fixed = 1.000·τ.
- T_ODE (fixed) examples: GIS 10000→3807, AMOC 50→19.0, WAIS 2000→761, AMAZ 100→38.1 yr.
- Time to x>0 in units of T_ODE: 15.1, 6.5, 3.4, 2.0, 1.1 (ratios 1.05, 1.2, 1.5, 2.0, 3.0).
- Example GIS τ=10000 at ratio 2.0: current 20174 yr, fixed 7680 yr.
- Crossings within the 12,700-yr committed window: current 43/60, fixed 51/60 (the fix lets slower elements tip in-window).
- Pipeline solver (LSODA, atol=rtol=1e-3) vs tight: max deviation 0.064% (<1%).
