import cv2
import sys

if len(sys.argv) != 3:
    print("Command should be of structure: python \{program name\} \{original image\} \{resulting image\}")
else:
    original = cv2.imread(sys.argv[1])
    result = cv2.imread(sys.argv[2])

    h_o, w_o, c_o = original.shape
    h_r, w_r, c_r = result.shape
    h_min = min(h_o, h_r)
    w_min = min(w_o, w_r)
    original2 = original
    result2 = result
    if (h_o != h_r or w_o != w_r):
        original2 = cv2.resize(original, (w_min, h_min), interpolation = cv2.INTER_AREA)
        result2 = cv2.resize(result, (w_min, h_min), interpolation = cv2.INTER_AREA)

    diff = cv2.absdiff(original2, result2)
    mse = diff * diff
    cv2.imshow('original', original2)
    cv2.imshow('result', result2)
    cv2.imshow('mse', mse)
    mse_num = sum(cv2.sumElems(mse)) / (h_min * w_min * 3)
    print("mean squared error: " + str(mse_num))
    cv2.imwrite(sys.argv[2].split(".")[0] + "_diff.png", diff)
    cv2.waitKey(0)

