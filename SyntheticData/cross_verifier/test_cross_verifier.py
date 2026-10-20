import math
import unittest
import run_cross_verifier as e


class CrossTests(unittest.TestCase):
    def test_log_domain_preserves_near_one_ordering(self):
        a=e.stable_log_probabilities(dict(World=0.0,Sports=-50.0,Business=-60.0,SciTech=-70.0))
        b=e.stable_log_probabilities(dict(World=0.0,Sports=-40.0,Business=-60.0,SciTech=-70.0))
        self.assertLess(a['World'],0)
        self.assertGreater(a['World'],b['World'])
        self.assertEqual(math.exp(a['World']),math.exp(b['World']))
        self.assertAlmostEqual(sum(math.exp(v) for v in a.values()),1)

    def test_self_cross_switch_verifier_and_balance(self):
        rows=[];score={s:{} for s in ['qwen','llama']}
        for source in ['qwen','llama']:
            for label in range(4):
                for i in range(50):
                    row=dict(source=source,label=label,sample_id=f'{source}_{label}_{i:02d}',proximity_score=i/50,text=str(i))
                    rows.append(row)
                    for verifier in score:
                        value=i if verifier==source else -i
                        score[verifier][row['sample_id']]=dict(stable_log_probability={e.repair.LABELS[label]:value})
        own=e.selected_dataset(rows,'SelfVerifier320',None,score)
        cross=e.selected_dataset(rows,'CrossVerifier320',None,score)
        self.assertEqual(len(own),320);self.assertEqual(len(cross),320)
        self.assertTrue(all(int(x['text'])>=10 for x in own))
        self.assertTrue(all(int(x['text'])<40 for x in cross))
        auxiliary=e.selected_dataset(rows,'CrossQwen160',None,score)
        self.assertEqual(len(auxiliary),160)
        self.assertEqual({x['source'] for x in auxiliary},{'qwen'})
        r1=e.selected_dataset(rows,'Random320',41)
        r2=e.selected_dataset(list(reversed(rows)),'Random320',41)
        self.assertEqual(r1,r2)

    def test_cache_matches_full_sequence_for_both_architectures(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
        from cached_scoring import candidate_values
        torch.set_num_threads(2);torch.manual_seed(4)
        configs=[(LlamaForCausalLM,LlamaConfig(vocab_size=40,hidden_size=24,intermediate_size=48,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)),(Qwen3ForCausalLM,Qwen3Config(vocab_size=40,hidden_size=24,intermediate_size=48,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=6,max_position_embeddings=64))]
        prefixes=[[3,4,5,6],[7,8]];suffixes=[[10,11],[12,11],[13,14,11],[15,16,17,11]]
        for klass,config in configs:
            model=klass(config).eval()
            actual=candidate_values(model,prefixes,suffixes,0,'cpu')
            expected=[]
            with torch.inference_mode():
                for prefix in prefixes:
                    for suffix in suffixes:
                        value=0
                        for i,token in enumerate(suffix):
                            logits=model(input_ids=torch.tensor([prefix+suffix[:i]]),use_cache=False,logits_to_keep=1).logits[0,-1]
                            value+=torch.log_softmax(logits.float(),dim=-1)[token].item()
                        expected.append(value)
            for a,b in zip(actual,expected):self.assertAlmostEqual(a,b,places=4)

    def test_random_pairing_averages_selection_seeds_first(self):
        import pandas as pd
        from analyze_cross_verifier import paired_frames,signal_class
        rows=[]
        shifts={'TopK320':.02,'SelfVerifier320':.01,'CrossVerifier320':.04,'SelfQwen160':0,'CrossQwen160':.01,'SelfLlama160':0,'CrossLlama160':.01}
        for method,selection,seed in e.training_specs():
            value=.7+.01*(seed-42)+(.001*(selection-43) if method=='Random320' else shifts[method])
            rows.append(dict(Method=method,Selection_Seed=selection,Model_Seed=seed,**{metric:value for metric in e.METRICS}))
        paired,summary,methods,random_mean=paired_frames(pd.DataFrame(rows))
        self.assertEqual(len(rows),60);self.assertEqual(len(paired),35)
        for _,x in random_mean.iterrows():self.assertAlmostEqual(x.Dev_Macro_F1,.7+.01*(x.Model_Seed-42))
        d=paired[paired.Comparison=='CrossVerifier320 - RandomMean'].Dev_Macro_F1_Delta
        self.assertTrue(all(abs(x-.04)<1e-12 for x in d))
        self.assertEqual(signal_class(summary),'Strong signal')
        row=(summary.Comparison=='CrossVerifier320 - SelfVerifier320')&(summary.Metric=='Dev_Macro_F1')
        summary.loc[row,'DeltaMean']=.007
        self.assertEqual(signal_class(summary),'Weak signal')
        summary.loc[row,'DeltaMean']=.001
        self.assertEqual(signal_class(summary),'No useful signal')


if __name__=='__main__':unittest.main()
