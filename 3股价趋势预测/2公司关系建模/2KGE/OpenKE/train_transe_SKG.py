import pickle

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import os, torch

datasets = "SKG"
# dataloader for training
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/%s/" % datasets,
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/%s/" % datasets, "link")

# define the model
transe = TransE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True)

# define the loss function
model = NegativeSampling(
    model=transe,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size()
)

# train the model
# trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
# trainer.run()

if not os.path.exists("./checkpoint"):
    os.mkdir("./checkpoint")

save_path = './checkpoint/%s/transe.ckpt' % datasets
if not os.path.exists(save_path):
    os.mkdir(save_path)
transe.save_checkpoint('./checkpoint/%s/transe.ckpt' % datasets)

# test the model
transe.load_checkpoint('./checkpoint/%s/transe.ckpt' % datasets)


# tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction(type_constrain = False)

def embedding2np(embeddings_layer):
    num_embeddings = embeddings_layer.num_embeddings
    embedding_dim = embeddings_layer.embedding_dim
    return transe.ent_embeddings(torch.LongTensor(list(range(num_embeddings)))).detach().cpu().numpy()


ent_embeddings = embedding2np(transe.ent_embeddings)
rel_embeddings = embedding2np(transe.rel_embeddings)
embedding = {"ent_embeddings": ent_embeddings, "rel_embeddings": rel_embeddings}
embedding_path = './checkpoint/%s/embedding_dict' % datasets
pickle.dump(embedding, open(embedding_path, 'wb'), protocol=3)

embedding = pickle.load(open(embedding_path, 'rb'))
