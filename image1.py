import cv2
import numpy as np
img = cv2.imread(r'C:\Users\jayra\Downloads\New folder\image copy 2.png')
if img is None:
    print("Error: input.jpg not found.")
    exit()

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11,11), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)
if contours:
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
else:
    mask[:] = 255

mask3 = cv2.merge([mask, mask, mask])
invmask3 = cv2.merge([cv2.bitwise_not(mask), cv2.bitwise_not(mask), cv2.bitwise_not(mask)])

subject = cv2.bitwise_and(enhanced, mask3)
subject_smooth = cv2.bilateralFilter(subject, d=11, sigmaColor=75, sigmaSpace=75)

bg = cv2.bitwise_and(enhanced, invmask3)
bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
bg_hsv[...,1] = (bg_hsv[...,1]*0.25).astype(np.uint8)
bg_desat = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2BGR)
bg_blur = cv2.GaussianBlur(bg_desat, (21,21), 0)

edges = cv2.Canny(enhanced, 100, 200)
edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edges_artistic = cv2.addWeighted(edges_col, 0.7, np.zeros_like(edges_col), 0.3, 0)

fmask = mask.astype(float)/255
inv_fmask = 1.0 - fmask
subj_final = subject_smooth.astype(float) * fmask[...,None]
bg_final = bg_blur.astype(float) * inv_fmask[...,None]
combined = cv2.add(subj_final, bg_final).astype(np.uint8)
stylized = cv2.addWeighted(combined, 1, edges_artistic, 0.25, 0)

mask_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
montage = np.hstack([
    cv2.resize(img, (256,256)),
    cv2.resize(mask_preview, (256,256)),
    cv2.resize(enhanced, (256,256)),
    cv2.resize(stylized, (256,256))
])

cv2.imshow('Montage: Original | Mask | Enhanced | Stylized', montage)
cv2.imwrite('stylized_result.jpg', stylized)
cv2.waitKey(0)
cv2.destroyAllWindows()
