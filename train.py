from FCTFT import *
# from option import opt
from self_option import opt
from self_utils import *
from torch.utils.data import DataLoader

# plt.show()


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

###########SSR_net       RGB---HSI  ---SSRl######################################333
opt.batch_size=10
opt.end_epoch=300

opt.outf='/data1/lxx_data/FCTFT/'

per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch

print("\nloading dataset ...")
#bgrs[3,482,512]    hypers:[31,482,512]
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.crop_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))


model=FCTFT().cuda().float()


############################################33
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_ssim=Loss_SSIM()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
# mse=torch.nn.L1Loss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

def main():
    iteration = 0
    record_mrae_loss = 1000
    # total_iteration=1200
    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_mrae(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration=i+1
            if iteration % 20 == 0:
                 print('algorithm:FCTFT: [iter:%d/%d],lr=%.9f,train_losses.avg=%.9f' % (iteration, total_iteration, lr, losses.avg))
            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss,ssim_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss},SSIM:{ssim_loss}')
                # Save model
                # if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                #     print(f'Saving to {opt.outf}')
                #     save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                #     if mrae_loss < record_mrae_loss:
                #         record_mrae_loss = mrae_loss
                # print loss
                if torch.abs(psnr_loss)>33.2:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf,(iteration//1000),psnr_loss,iteration,model,optimizer)
                ################################
                ###################################
                #########################################
                print(" Epoch[%06d],Iter[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f  Test SSIM: %.9f" % (iteration,iteration/1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss,ssim_loss))

    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ssim=AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_ssim=criterion_ssim(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_ssim.update(loss_ssim.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg,loss_ssim

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()
    # torch.save(model.state_dict(), PATH6)
    print(torch.__version__)