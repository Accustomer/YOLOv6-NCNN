# YOLOv6-NCNN
Deploy YOLOv6 by NCNN

## 1. Change Detect module
Change the forward function of Detect module (In effidehead*.py) as follow:
```
def forward(self, x):
    if self.training:
        ...(Do not change)
    else: 
        cls_score_list = []
        reg_lrtb_list = []
        for i in range(self.nl):
            b, _, h, w = x[i].shape
            l = h * w
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output_lrtb = self.reg_preds_lrtb[i](reg_feat)
                        
            cls_output = torch.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([b, self.nc, l]))
            reg_lrtb_list.append(reg_output_lrtb.reshape([b, 4, l]))
        
        cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
        reg_lrtb_list = torch.cat(reg_lrtb_list, axis=-1).permute(0, 2, 1)
        
        # Disable the part of parsing bbox
        EXPORT_NCNN = True
        if EXPORT_NCNN:
            return torch.cat([
                    reg_lrtb_list,
                    torch.ones((b, reg_lrtb_list.shape[1], 1), device=reg_lrtb_list.device, dtype=reg_lrtb_list.dtype),
                    cls_score_list], 
                    axis=-1
                )

        anchor_points, stride_tensor = generate_anchors(
            x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')
        pred_bboxes = dist2bbox(reg_lrtb_list, anchor_points, box_format='xywh')
        pred_bboxes *= stride_tensor
        return torch.cat(
            [
                pred_bboxes,
                torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                cls_score_list
            ],
            axis=-1)
```
Set the **EXPORT_NCNN** to **True** before export.

## 2. Export to ONNX
Use following commands to export Pytorch model to ONNX model:
```
cd YOLOv6
python deploy/ONNX/export_onnx.py --weights path-to-pt-model --img-size 640 640 --batch-size 1 --simplify --device 0
```
**--simplify** must be required, the othe parameters can be changed according to the actual situation.

## 3. Convert ONNX to NCNN
Use the executable file "onnx2ncnn.exe" to convert ONNX model to NCNN model:
```
path-to-onnx2ncnn.exe path-to-yolov6.onnx path-to-yolov6.param path-to-yolov6.bin
```
Then, modify the first dimension of **Reshape** layer in .param file:
```
# Before
...
Reshape          Reshape_143              1 1 290 303 0=6400 1=17 2=4
...
Reshape          Reshape_153              1 1 307 316 0=6400 1=80
Reshape          Reshape_159              1 1 306 325 0=6400 1=4
...
Reshape          Reshape_184              1 1 346 359 0=1600 1=17 2=4
...
Reshape          Reshape_194              1 1 363 372 0=1600 1=80
Reshape          Reshape_200              1 1 362 381 0=1600 1=4
...
Reshape          Reshape_225              1 1 402 415 0=400 1=17 2=4
...
Reshape          Reshape_235              1 1 419 428 0=400 1=80
Reshape          Reshape_241              1 1 418 437 0=400 1=4
...

# After
...
Reshape          Reshape_143              1 1 290 303 0=-1 1=17 2=4
...
Reshape          Reshape_153              1 1 307 316 0=-1 1=80
Reshape          Reshape_159              1 1 306 325 0=-1 1=4
...
Reshape          Reshape_184              1 1 346 359 0=-1 1=17 2=4
...
Reshape          Reshape_194              1 1 363 372 0=-1 1=80
Reshape          Reshape_200              1 1 362 381 0=-1 1=4
...
Reshape          Reshape_225              1 1 402 415 0=-1 1=17 2=4
...
Reshape          Reshape_235              1 1 419 428 0=-1 1=80
Reshape          Reshape_241              1 1 418 437 0=-1 1=4
...
```
Otherwise, you can use fp16 to save and inference model by this command:
```
path-to-ncnnoptimize.exe path-to-yolov6.param path-to-yolov6.bin path-to-yolov6-opt.param path-to-yolov6-opt.bin 65536
```

## 4. Deploy NCNN model
Load and inference NCNN models as the way in **yolov6.cpp**. Here is an example of detection result:
![000000000597-mark](https://github.com/Accustomer/YOLOv6-NCNN/blob/main/images/000000000597-mark.jpg)


