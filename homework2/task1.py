f = 50
N = 1.8
c = 5 * 36 / 5616
s1 = 2500
M = f / (s1 - f)

N1 = 3
N2 = 16
for item in [N1, N2]:
    D = f / item
    d_far = M * D * s1 / (M * D - c)
    d_close = M * D * s1 / (M * D + c)
    DoF1 = d_far - d_close
    print('when the aperture is f/' + str(item))
    print('the near distance is ', round(d_close, 2), 'mm')
    print('the far distance is ', round(d_far, 2), 'mm')
    print('the DoF is ', round(DoF1, 2), 'mm')
print('____________________________________________________')

S1 = 500
S2 = 20000
for item in [S1, S2]:
    D = f / 3
    M=f / (item - f)
    d_far = M * D * item / (M * D - c)
    d_close = M * D * item / (M * D + c)
    DoF1 = d_far - d_close
    print('when the focus distance is ' + str(item) + 'mm')
    print('the near distance is ', round(d_close, 2), 'mm')
    print('the far distance is ', round(d_far, 2), 'mm')
    print('the DoF is ', round(DoF1, 2), 'mm')
