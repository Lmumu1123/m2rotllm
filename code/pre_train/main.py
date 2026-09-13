import torch
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dotenv import dotenv_values

from code.models.SFN import SpecFoldNet
from code.pre_train.dataloader import split_file_list, VibrationDataset
from code.pre_train.dataloader import get_filtered_sample_list
# from code.exp.gener_study import get_filtered_sample_list

torch.set_float32_matmul_precision('high')


class LtModel(L.LightningModule):
    def __init__(self, **kwargs):
        super(LtModel, self).__init__()
        fold_num = kwargs.get('fold_num', 3)
        self.model = SpecFoldNet(fold_num=fold_num)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = cross_entropy(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc, sync_dist=True, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc, sync_dist=True)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               'min',
                                                               patience=150,
                                                               factor=0.5,
                                                               threshold=1e-3)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'strict': False,
                    'monitor': 'val_loss'
                }}


def get_dataloader(batch_size=2000, dataset_filter='', snr=None):
    train_set, val_set, test_set = split_file_list(get_filtered_sample_list(filter_str=dataset_filter))
    train_loader = DataLoader(VibrationDataset(train_set, snr=snr), batch_size=batch_size, shuffle=True,
                              num_workers=12, drop_last=False)
    val_loader = DataLoader(VibrationDataset(val_set), batch_size=batch_size, shuffle=False,
                            num_workers=12, drop_last=False)
    test_loader = DataLoader(VibrationDataset(test_set), batch_size=batch_size, shuffle=False,
                             num_workers=12, drop_last=False)
    return train_loader, val_loader, test_loader


def train_model(**kwargs):
    batch_size = kwargs.get('batch_size', 2000)
    max_epochs = kwargs.get('max_epochs', 50)
    dataset_filter = kwargs.get('dataset_filter', '')

    train_loader, val_loader, test_loader = get_dataloader(batch_size=batch_size, dataset_filter=dataset_filter)
    auto_model = LtModel(**kwargs)
    trainer = L.Trainer(max_epochs=max_epochs,
                        default_root_dir=dotenv_values()['LOG_PATH'],
                        benchmark=True,
                        log_every_n_steps=10,
                        callbacks=[ModelCheckpoint(monitor='val_loss', mode='min', filename='best_loss'),
                                   ModelCheckpoint(monitor='val_acc', mode='max', filename='best_acc'),
                                   EarlyStopping(monitor='val_loss', min_delta=1e-4, mode="min", verbose=True)], )
    trainer.fit(auto_model, train_loader, val_loader)
    log_dir = trainer.log_dir

    @rank_zero_only
    def test_model():
        m = LtModel.load_from_checkpoint(log_dir + '/checkpoints/best_loss.ckpt')
        t = L.Trainer(devices=1, logger=False)
        test_acc = t.test(m, test_loader)[0]['test_acc']
        open(f'{log_dir}/test_acc.txt', 'w').write(str(test_acc))

    test_model()


if __name__ == "__main__":
    train_model()
