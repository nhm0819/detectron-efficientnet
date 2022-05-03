import torch
from detectron2.engine import DefaultPredictor
from detectron2.modeling.meta_arch.build import build_model
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer


class PostProcess(torch.nn.Module):
    @torch.no_grad()
    def forward(self, outputs, original_image, mask_id=25):
        masks = []
        wire_probs = []
        for idx, output in enumerate(outputs):
            instances = output["instances"] # .to("cpu")
            # img = original_image[idx]["image"].clone().type(torch.uint8).permute(1, 2, 0).numpy()
            img = original_image[idx]["image"].permute(1, 2, 0)
            wire_idxs = (instances.pred_classes == mask_id).nonzero().squeeze()

            if wire_idxs.nelement() == 0:
                masks.append(img.numpy())
                wire_probs.append(0)
                continue

            elif wire_idxs.nelement() == 1:
                wire_idx = wire_idxs.item()

            else:
                wire_idx = wire_idxs.to('cpu').numpy()[0]

            wire_prob = instances.scores[wire_idx].item()
            mask = torch.as_tensor(instances.pred_masks[wire_idx], device='cpu')     # .clone().detach()
            
            # img_cuda = img.to('cuda:0')

            mask_gt = torch.gt(mask, 0) # np.greater(mask, 0)  # get only non-zero positive pixels/labels
            mask_exp = mask_gt.unsqueeze(-1) # np.expand_dims(mask, axis=-1)  # (H, W) -> (H, W, 1)
            mask_3c = torch.cat((mask_exp, mask_exp, mask_exp), axis=-1) # np.concatenate((mask, mask, mask), axis=-1)
            
            # mask_mul = torch.mul(img_cuda, mask_3c) # np.multiply(img, mask)
            mask_mul = torch.mul(img, mask_3c)
            
            # mask_image = torch.where(mask_mul == 0, torch.tensor(255, dtype=torch.uint8, device='cuda'), mask_mul) # np.where(mask_image == 0)
            mask_image = torch.where(mask_mul==0, torch.tensor(255, dtype=torch.uint8), mask_mul)
            # mask_image[where_0] = 255

            try:
                x1, y1, x2, y2 = instances.pred_boxes.tensor[0].type(torch.int64).squeeze() # output.pred_boxes.tensor[wire_idx].squeeze().numpy().astype(int)
                mask_crop = mask_image[y1:y2, x1:x2]
            except:
                mask_crop = mask_image

            # masks.append(mask_crop.to("cpu").clone().detach().numpy())
            masks.append(mask_crop.numpy())
            
            wire_probs.append(wire_prob)
            
            # import matplotlib.pyplot as plt
            # plt.imshow(mask_image[:, :, 0].to('cpu').numpy().astype(int))

        return masks, wire_probs


class CustomPredictor(DefaultPredictor):
    def __init__(self, cfg, mask_id=25):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.mask_id = mask_id

        self.post_process = PostProcess()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format


    def __call__(self, original_image):

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            predictions = self.model(original_image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            masks, wire_probs = self.post_process(predictions, original_image, mask_id=self.mask_id)

            return masks, wire_probs
