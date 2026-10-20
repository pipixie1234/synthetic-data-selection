import json
import unittest

import run_repair as e


class RepairTests(unittest.TestCase):
    def test_same200_and_anchor_disjointness(self):
        samples=e.old.read_jsonl(e.OUT/'samples.jsonl')
        previous=[x for x in e.old.read_jsonl(e.PREVIOUS/'samples.jsonl') if x['subset']=='main']
        self.assertEqual(samples,previous)
        anchors=e.old.read_jsonl(e.OUT/'real_anchors.jsonl')
        self.assertEqual(len(anchors),8)
        self.assertFalse({x['original_id'] for x in samples}&{x['original_id'] for x in anchors})
        self.assertFalse({e.norm(x['text']) for x in samples}&{e.norm(x['text']) for x in anchors})
        self.assertEqual({label:sum(x['true_label']==label for x in anchors) for label in e.LABELS},dict.fromkeys(e.LABELS,2))

    def test_label_prompt_does_not_use_presented_label(self):
        anchors=e.old.read_jsonl(e.OUT/'real_anchors.jsonl')
        samples=e.old.read_jsonl(e.OUT/'samples.jsonl')
        for mode in e.MODES:
            prompts=e.old.read_jsonl(e.OUT/f'{mode}_label_prompts.jsonl')
            for x,p in zip(samples,prompts):
                self.assertEqual(p['prompt'],e.label_prompt(x['text'],anchors if mode=='real_anchored' else []))
                self.assertNotIn('Presented label:',p['prompt'])
                self.assertNotIn(x['item_id'],p['prompt'])
                self.assertNotIn(x['original_id'],p['prompt'])

    def test_quality_task_is_independent(self):
        samples=e.old.read_jsonl(e.OUT/'samples.jsonl')
        prompts=e.old.read_jsonl(e.OUT/'quality_prompts.jsonl')
        for x,p in zip(samples,prompts):
            self.assertEqual(p['prompt'],e.QUALITY_PROMPT.format(text=json.dumps(x['text'],ensure_ascii=False)))
            self.assertNotIn('Presented label:',p['prompt'])
            self.assertNotIn(x['item_id'],p['prompt'])

    def test_code_pass_cannot_accept_wrong_label(self):
        for news in [False,True]:
            for plausible in [False,True]:
                q=dict(news_like=news,plausible=plausible,coherent=False,template_like=True)
                self.assertEqual(e.make_pass('Sports','World',q),(False,False))
                self.assertEqual(e.make_pass('Sports','Sports',q),(True,news and plausible))

    def test_parsers_do_not_impute(self):
        self.assertEqual(e.parse_label('predicted_label: SciTech\n'),{'predicted_label':'SciTech'})
        q=e.parse_quality('news_like: yes\ncoherent: no\nplausible: yes\ntemplate_like: no')
        self.assertEqual(q,dict(news_like=True,coherent=False,plausible=True,template_like=False))
        for raw in ['predicted_label: World\nPASS','World','predicted_label: Sci/Tech']:
            with self.assertRaises(ValueError): e.parse_label(raw)
        with self.assertRaises(ValueError): e.parse_quality('news_like: maybe\ncoherent: no\nplausible: yes\ntemplate_like: no')

    def test_logprob_batch_matches_sequential_next_token_reference(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(7)
        model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=24,intermediate_size=48,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.0)).eval()
        prefixes=[[3,4,5,6],[7,8],[3,4,5,6]]
        suffixes=[[10,11],[12,13,14],[15,11]]
        sequences=[p+s for p,s in zip(prefixes,suffixes)]
        length=max(map(len,sequences))
        ids=torch.tensor([[0]*(length-len(x))+x for x in sequences])
        attention=torch.tensor([[0]*(length-len(x))+[1]*len(x) for x in sequences])
        positions=(attention.cumsum(-1)-1).masked_fill(attention==0,0)
        with torch.inference_mode():
            output=model(input_ids=ids,attention_mask=attention,position_ids=positions,use_cache=False,logits_to_keep=4)
            batched=e.suffix_log_probabilities(output.logits,suffixes)
            reference=[]
            for prefix,suffix in zip(prefixes,suffixes):
                value=0.0
                for i,token in enumerate(suffix):
                    logits=model(input_ids=torch.tensor([prefix+suffix[:i]]),use_cache=False,logits_to_keep=1).logits[0,-1]
                    value+=torch.log_softmax(logits.float(),dim=-1)[token].item()
                reference.append(value)
        for actual,expected in zip(batched,reference): self.assertAlmostEqual(actual,expected,places=5)

    def test_native_chat_has_only_one_bos(self):
        from transformers import AutoTokenizer
        for name in ['Qwen3-4B-Instruct-2507','Llama-3.2-3B-Instruct']:
            tokenizer=AutoTokenizer.from_pretrained(e.ROOT/'cloud/model'/name,local_files_only=True)
            messages=[dict(role='user',content=e.label_prompt('A football team won the final.',[]))]
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            native=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)['input_ids']
            explicit=tokenizer(rendered,add_special_tokens=False).input_ids
            self.assertEqual(explicit,native)
            if name.startswith('Llama'):
                self.assertEqual(explicit.count(tokenizer.bos_token_id),1)


if __name__=='__main__': unittest.main()
