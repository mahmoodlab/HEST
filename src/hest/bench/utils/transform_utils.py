from torchvision import transforms

def get_eval_transforms(mean, std, target_img_size = -1, center_crop = False):
	trsforms = []
	
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
		if center_crop:
			trsforms.append(transforms.CenterCrop(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms