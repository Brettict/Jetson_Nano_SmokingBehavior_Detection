import cv2
import numpy as np
import paddle.inference as paddle_infer
from paddle.inference import Config
from paddle.inference import create_predictor


def resize(img, target_size):
    """resize to target size"""
    if not isinstance(img, np.ndarray):
        raise TypeError('image type is not numpy.')
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale_x = float(target_size) / float(im_shape[1])
    im_scale_y = float(target_size) / float(im_shape[0])
    img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y)
    return img


def normalize(img, mean, std):
    img = img / 255.0
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std
    return img


def preprocess(img, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = resize(img, img_size)
    img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
    img = normalize(img, mean, std)
    img = img.transpose((2, 0, 1))  # hwc -> chw
    return img[np.newaxis, :]

#########进行模型配置###################################

def Config(prog_file,params_file):
    # 创建 config
    config = paddle_infer.Config()

    # 通过 API 设置模型文件夹路径
    #config.set_prog_file("./mobilenet_v2/__model__")
    #config.set_params_file("./mobilenet_v2/__params__")
    config.set_prog_file(prog_file)
    config.set_params_file(params_file)

    # 根据 config 创建 predictor
    config.enable_use_gpu(1000, 0)
    config.switch_ir_optim()
    config.enable_memory_optim()
    #config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=paddle_infer.PrecisionType.Float32,max_batch_size=1, min_subgraph_size=5, use_static=False, use_calib_mode=False)

    predictor = paddle_infer.create_predictor(config)

    return predictor

def predic(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results

def draw_bbox(img, result,label_list, threshold=0.35): #threshold =0.5 original scores
	
    """draw bbox"""
    #text = "Alert"
    for res in result:
        id, score, bbox = res[0], res[1], res[2:]
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (255,0,255), 2)
#cv2.rectangle(img, (int(xmin-30),int(ymin-50)), (int(xmax+50), int(ymax+50)), (255,0,0), 5)
        print('category id is {}, bbox is {}'.format(id, bbox))
        try:
            label_id = label_list[int(id)]
            cv2.putText(img, label_id, (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.putText(img, str(round(score,2)), (int(xmin-35), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)	        
            #cv2.putText(img, str(round(score,2)), (int(xmin-35), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        except KeyError:
            pass
if __name__ == '__main__':
    label_list="home/yuan/data/voc/labels.txt"
    print("开始进行预测")
    #摄像头读取
    #path = "home/yuan/PaddleDetection/demo_yuan/smoke.mp4"
    cap = cv2.VideoCapture("demo_yuan/smoke.mp4")
    # 图像尺寸相关参数初始化
    ret, img = cap.read()
    #模型文件路径
    prog_file = './inference_model/yolov3_mobilenet_v1_270e_voc/model.pdmodel'
    params_file = './inference_model/yolov3_mobilenet_v1_270e_voc/model.pdiparams'
    predictor = Config(prog_file,params_file)
    img_size = 224
    scale_factor = np.array([img_size * 1. / img.shape[0], img_size * 1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
    im_shape = np.array([img_size, img_size]).reshape((1, 2)).astype(np.float32)
    while True:
    #while cap.isOpened():
        ret, img = cap.read()
        pro_data = preprocess(img,img_size)
        result = predic(predictor, [im_shape, pro_data, scale_factor])
        draw_bbox(img,result[0],label_list)
        cv2.imshow("pred", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #cap.release()
    #cv2.destroyAllWindows()
