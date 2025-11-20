# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main Retriever class
# --------------------------------------------------------
import os
import argparse
import numpy as np
import torch

from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.model import RetrievalModel, extract_local_features

try:
    import faiss
    faiss.StandardGpuResources()  # when loading the checkpoint, it will try to instanciate FaissGpuL2Index
except AttributeError as e:
    import asmk.index

    class FaissCpuL2Index(asmk.index.FaissL2Index):
        def __init__(self, gpu_id):
            super().__init__()
            self.gpu_id = gpu_id

        def _faiss_index_flat(self, dim):
            """Return initialized faiss.IndexFlatL2"""
            return faiss.IndexFlatL2(dim)

    asmk.index.FaissGpuL2Index = FaissCpuL2Index

from asmk import asmk_method  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser('Retrieval scores from a set of retrieval', add_help=False, allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help="shortname of a retrieval model or path to the corresponding .pth")
    parser.add_argument('--input', type=str, required=True,
                        help="directory containing images or a file containing a list of image paths")
    parser.add_argument('--outfile', type=str, required=True, help="numpy file where to store the matrix score")
    return parser


def get_impaths(imlistfile):
    with open(imlistfile, 'r') as fid:
        impaths = [f for f in imlistfile.read().splitlines() if not f.startswith('#')
                   and len(f) > 0]  # ignore comments and empty lines
    return impaths


def get_impaths_from_imdir(imdir, extensions=['png', 'jpg', 'PNG', 'JPG']):
    assert os.path.isdir(imdir)
    impaths = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if any(f.endswith(ext) for ext in extensions)]
    return impaths


def get_impaths_from_imdir_or_imlistfile(input_imdir_or_imlistfile):
    if os.path.isfile(input_imdir_or_imlistfile):
        return get_impaths(input_imdir_or_imlistfile)
    else:
        return get_impaths_from_imdir(input_imdir_or_imlistfile)


class Retriever(object):
    def __init__(self, modelname, backbone=None, device='cuda'):
        # load the model
        assert os.path.isfile(modelname), modelname
        print(f'Loading retrieval model from {modelname}')
        ckpt = torch.load(modelname, 'cpu', weights_only=False)  # TODO from pretrained to download it automatically
        ckpt_args = ckpt['args']
        if backbone is None:
            backbone = AsymmetricMASt3R.from_pretrained(ckpt_args.pretrained)
        self.model = RetrievalModel(
            backbone, freeze_backbone=ckpt_args.freeze_backbone, prewhiten=ckpt_args.prewhiten,
            hdims=list(map(int, ckpt_args.hdims.split('_'))) if len(ckpt_args.hdims) > 0 else "",
            residual=getattr(ckpt_args, 'residual', False), postwhiten=ckpt_args.postwhiten,
            featweights=ckpt_args.featweights, nfeat=ckpt_args.nfeat
        ).to(device)
        self.device = device
        msg = self.model.load_state_dict(ckpt['model'], strict=False)
        assert all(k.startswith('backbone') for k in msg.missing_keys)
        assert len(msg.unexpected_keys) == 0
        self.imsize = ckpt_args.imsize

        # load the asmk codebook
        dname, bname = os.path.split(modelname)  # TODO they should both be in the same file ?
        bname_splits = bname.split('_')
        cache_codebook_fname = os.path.join(dname, '_'.join(bname_splits[:-1]) + '_codebook.pkl')
        assert os.path.isfile(cache_codebook_fname), cache_codebook_fname
        asmk_params = {'index': {'gpu_id': 0}, 'train_codebook': {'codebook': {'size': '64k'}},
                       'build_ivf': {'kernel': {'binary': True}, 'ivf': {'use_idf': False},
                                     'quantize': {'multiple_assignment': 1}, 'aggregate': {}},
                       'query_ivf': {'quantize': {'multiple_assignment': 5}, 'aggregate': {},
                                     'search': {'topk': None},
                                     'similarity': {'similarity_threshold': 0.0, 'alpha': 3.0}}}
        asmk_params['train_codebook']['codebook']['size'] = ckpt_args.nclusters
        self.asmk = asmk_method.ASMKMethod.initialize_untrained(asmk_params)
        self.asmk = self.asmk.train_codebook(None, cache_path=cache_codebook_fname)

    def __call__(self, input_imdir_or_imlistfile, outfile=None):
        # get impaths
        if isinstance(input_imdir_or_imlistfile, str):
            impaths = get_impaths_from_imdir_or_imlistfile(input_imdir_or_imlistfile)
        else:
            impaths = input_imdir_or_imlistfile  # we're assuming a list has been passed
        print(f'Found {len(impaths)} images')

        # build the database
        feat, ids = extract_local_features(self.model, impaths, self.imsize, tocpu=True, device=self.device)
        feat = feat.cpu().numpy()
        ids = ids.cpu().numpy()
        asmk_dataset = self.asmk.build_ivf(feat, ids)

        # we actually retrieve the same set of images
        metadata, query_ids, ranks, ranked_scores = asmk_dataset.query_ivf(feat, ids)

        # well ... scores are actually reordered according to ranks ...
        # so we redo it the other way around...
        scores = np.empty_like(ranked_scores)
        scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores

        # save
        if outfile is not None:
            if os.path.isdir(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
            np.save(outfile, scores)
            print(f'Scores matrix saved in {outfile}')
        return scores
