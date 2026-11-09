"""Subscription transport must preserve tools and never accept a partial SSE answer."""
import json
from unittest.mock import Mock

import httpx
import pytest
from somm.errors import SommRateLimited, SommUpstream5xx
from somm.providers.chatgpt import ChatGPTProvider, response_input


def test_tool_reasoning_round_trip():
    from somm.providers.chatgpt import chat_completion

    item = {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"}
    result = chat_completion({"id": "resp1", "model": "actual-model", "output": [item,
        {"type": "function_call", "call_id": "c1", "name": "groceries", "arguments": '{"item":"milk"}'}]})
    message = result["choices"][0]["message"]
    replay = response_input([message, {"role": "tool", "tool_call_id": "c1", "content": "added"}])
    assert replay == [item, {"type": "function_call", "call_id": "c1", "name": "groceries",
                            "arguments": '{"item":"milk"}'},
                      {"type": "function_call_output", "call_id": "c1", "output": "added"}]
    assert result["model"] == "actual-model"


def transport(monkeypatch, text, status=200):
    original = httpx.Client
    def handler(request):
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer synthetic"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(status, text=text)
    monkeypatch.setattr("somm.providers.chatgpt.httpx.Client", lambda **kw: original(
        transport=httpx.MockTransport(handler), **kw))
    return ChatGPTProvider(auth=Mock(access_token=Mock(return_value="synthetic")))


def test_only_completed_event_is_success_even_without_trailing_blank_line(monkeypatch):
    value = {"type": "response.completed", "response": {"status": "completed", "id": "r1",
             "model": "served-model", "output": [{"type": "message", "content": [
                 {"type": "output_text", "text": "hello"}]}], "usage": {"input_tokens": 2, "output_tokens": 1}}}
    provider = transport(monkeypatch, "data: " + json.dumps(value))
    result = provider.complete_chat({"model": "alias", "messages": [{"role": "user", "content": "hi"}]})
    assert result["model"] == "served-model"
    assert result["choices"][0]["message"]["content"] == "hello"


@pytest.mark.parametrize("event", ["response.failed", "response.incomplete", "response.output_text.delta"])
def test_http_200_does_not_make_failed_or_interrupted_response_success(monkeypatch, event):
    provider = transport(monkeypatch, 'data: ' + json.dumps({"type": event}) + '\n\n')
    with pytest.raises(SommUpstream5xx):
        provider.complete_chat({"model": "x", "messages": []})


def test_rate_limit_is_not_a_completed_answer(monkeypatch):
    provider = transport(monkeypatch, "", 429)
    with pytest.raises(SommRateLimited):
        provider.complete_chat({"model": "x", "messages": []})


def test_siwc_preview_uses_developer_and_additional_tools_without_output_cap(monkeypatch):
    original = httpx.Client
    def handler(request):
        body = json.loads(request.content)
        assert 'max_output_tokens' not in body and 'tools' not in body
        assert body['input'][0]['type'] == 'additional_tools'
        assert body['input'][0]['role'] == 'developer'
        assert body['input'][0]['tools'][0]['name'] == 'lookup'
        assert body['input'][1] == {'role': 'developer', 'content': 'Be accurate'}
        value = {'type': 'response.completed', 'response': {'status':'completed','model':'served','id':'r1','output':[]}}
        return httpx.Response(200, text='data: '+json.dumps(value)+'\n\n')
    monkeypatch.setattr('somm.providers.chatgpt.httpx.Client', lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    provider = ChatGPTProvider(auth=Mock(access_token=Mock(return_value='synthetic')))
    provider.complete_chat({'model':'x','messages':[{'role':'system','content':'Be accurate'}],
                            'max_tokens':4096,'tools':[{'type':'function','function':{'name':'lookup'}}]})


def test_replay_preserves_message_phase_and_original_item_order():
    from somm.providers.chatgpt import chat_completion
    items = [{'type':'reasoning','id':'r1','encrypted_content':'opaque'},
             {'type':'message','role':'assistant','phase':'commentary','content':[{'type':'output_text','text':'Looking it up'}]},
             {'type':'function_call','id':'fc1','call_id':'c1','name':'lookup','arguments':'{}'}]
    message = chat_completion({'id':'resp1','model':'served','output':items})['choices'][0]['message']
    assert response_input([message, {'role':'tool','tool_call_id':'c1','content':'found'}]) == [*items, {'type':'function_call_output','call_id':'c1','output':'found'}]


def expanded_batch():
    calls = [{'id':'parent' if i==0 else f'child_{i}', 'type':'function',
              'function':{'name':'tool_call','arguments':json.dumps({'calls':[{'tool':'lookup','arguments':{'i':i}}]})}}
             for i in range(3)]
    raw = [{'type':'reasoning','id':'r1','encrypted_content':'stale-prefix'},
           {'type':'message','role':'assistant','phase':'commentary','content':[{'type':'output_text','text':'Reading fixtures'}]},
           {'type':'function_call','call_id':'parent','name':'tool_call','arguments':json.dumps({'calls':[{'tool':'lookup','arguments':{'i':i}} for i in range(3)]})}]
    message = {'role':'assistant','content':'Reading fixtures','tool_calls':calls,
               'reasoning_details':[{'type':'somm.native_assistant','item':item} for item in raw]}
    return [{'role':'user','content':'Read these fixtures'},message,
            *[{'role':'tool','tool_call_id':call['id'],'content':f'completed-{i}'} for i,call in enumerate(calls)]]


def test_expanded_batch_recovers_from_completed_results_without_reexecuting_tools():
    from somm.providers.chatgpt import prepare_response_input
    messages = expanded_batch()
    items, recovery = prepare_response_input(messages)
    calls = [v for v in items if v.get('type')=='function_call']
    outputs = [v for v in items if v.get('type')=='function_call_output']
    assert [v['call_id'] for v in calls] == ['parent','child_1','child_2']
    assert [v['call_id'] for v in outputs] == ['parent','child_1','child_2']
    assert [v['output'] for v in outputs] == ['completed-0','completed-1','completed-2']
    assert not any(v.get('type')=='reasoning' for v in items)
    assert next(v for v in items if v.get('role')=='assistant')['phase']=='commentary'
    assert recovery == {'kind':'normalized_tool_replay','provider_retries':0,'tool_results_preserved':3}
    # The canonical history remains intact; recovery is a request-local projection.
    assert messages[1]['reasoning_details'][0]['item']['encrypted_content']=='stale-prefix'


def test_unrecoverable_orphan_is_fatal_before_auth_or_network():
    from somm.errors import SommBadRequest
    auth = Mock()
    with pytest.raises(SommBadRequest, match='no matching'):
        ChatGPTProvider(auth=auth).complete_chat({'model':'x','messages':[{'role':'tool','tool_call_id':'missing','content':'done'}]})
    auth.access_token.assert_not_called()


def test_null_code_invalid_request_is_not_retried_as_an_outage(monkeypatch):
    from somm.errors import SommBadRequest
    event={'type':'error','error':{'type':'invalid_request_error','code':None,'param':'input',
                                 'message':'No tool call found for function call output'}}
    provider=transport(monkeypatch,'data: '+json.dumps(event)+'\n\n')
    with pytest.raises(SommBadRequest, match='invalid_request_error') as failure:
        provider.complete_chat({'model':'x','messages':[]})
    assert failure.value.provider_code is None
    assert failure.value.provider_type=='invalid_request_error'
    assert failure.value.provider_param=='input'


def test_repaired_request_keeps_model_and_reports_recovery(monkeypatch):
    original=httpx.Client
    def handler(request):
        body=json.loads(request.content)
        assert body['model']=='strong-model'
        assert [v['call_id'] for v in body['input'] if v.get('type')=='function_call_output']==['parent','child_1','child_2']
        assert not any(v.get('type')=='reasoning' for v in body['input'])
        event={'type':'response.completed','response':{'id':'r1','model':'strong-model','status':'completed','output':[]}}
        return httpx.Response(200,text='data: '+json.dumps(event)+'\n\n')
    monkeypatch.setattr('somm.providers.chatgpt.httpx.Client',lambda **kw: original(transport=httpx.MockTransport(handler),**kw))
    result=ChatGPTProvider(auth=Mock(access_token=Mock(return_value='synthetic'))).complete_chat({'model':'strong-model','messages':expanded_batch()})
    assert result['model']=='strong-model'
    assert result['somm_recovery']['tool_results_preserved']==3
