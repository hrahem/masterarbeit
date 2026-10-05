"""Audit and pseudonymize this HubSpot batch. Standard library only.

Usage: python build_campaign_package.py INPUT_DIRECTORY OUTPUT_DIRECTORY
Optional HUBSPOT_PSEUDONYM_KEY: a private, high-entropy key for reproducible IDs.
Without it, a fresh in-memory key is generated; no identity map or key is exported.
"""
import os, sys, csv, json, re, hmac, hashlib, secrets, math, bisect
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from html.parser import HTMLParser

ROOT, OUT = map(Path, sys.argv[1:3])
OUT.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get('HUBSPOT_PSEUDONYM_KEY', '').encode() or secrets.token_bytes(32)
OBSERVATION_END = datetime(2026, 9, 21, 0, 0)  # conservative export-day boundary, source-local clock
EMAIL = re.compile(r'[A-Za-z0-9.!#$%&\x27*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+')

def token(kind, value):
    if not value: return ''
    return kind + '_' + hmac.new(KEY, (kind+'\0'+value).encode(), hashlib.sha256).hexdigest()[:24]

def dt(value):
    if not value: return None
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')

def norm(s): return ' '.join(s.split())
def write_csv(name, rows):
    if not rows: return
    path=OUT/name;path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class EmailHTML(HTMLParser):
    def __init__(self):
        super().__init__();self.in_title=False;self.title='';self.skip=0;self.parts=[];self.images=0;self.links=0
    def handle_starttag(self,t,attrs):
        if t=='title':self.in_title=True
        if t in ('style','script'):self.skip+=1
        if t=='img':self.images+=1
        if t=='a':self.links+=1
    def handle_endtag(self,t):
        if t=='title':self.in_title=False
        if t in ('style','script'):self.skip=max(0,self.skip-1)
    def handle_data(self,s):
        if self.in_title:self.title+=s
        if not self.skip and not self.in_title:
            s=norm(s.replace('\u034f','').replace('\xad',''))
            if s:self.parts.append(s)

# Manually transcribed from supplied screenshots. Values describe the visible reports,
# not a reconstructed recipient-level bot classification or a proof of randomization.
SCREEN = {
 'DACH_DE_BASIC_checkoutfeb25_2A': [393,393,60,6,'on',0,'even_split',''],
 'DE_NL-CW12-25_DACH-DE-LEADSvA': [2521,2504,502,48,'on',0,'even_split',''],
 'DE_NL-CW12-25_DACH-DE-PAIDvA': [5651,5635,1222,126,'on',0,'even_split',''],
 'DE_NL-CW12-25_ROW-DE-PAIDvA': [185,183,50,7,'on',0,'even_split',''],
 'DE_NL-CW20-23_DACH-DE-LEAD_A': [1639,1635,427,99,'unavailable',4,'inconclusive_fallback','A'],
 'NL_Webinar_DACH-DE-LEADs_A': [2384,2375,457,59,'on',0,'even_split',''],
 'dach_de_free_business26_4': [7505,7495,1629,27,'on',2,'reported_winner','B'],
 'dach_de_plus_m_consistency26m': [196,196,22,3,'on',2,'inconclusive_fallback','A'],
 'dach_de_plus_m_consistency26m_2': [199,199,28,2,'on',2,'reported_winner','B'],
 'dach_de_plus_m_consistency26m_3': [196,196,22,3,'on',2,'inconclusive_fallback','A'],
}
raw={};html={};signatures={};canonical={};issues=[]
for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    files=list(folder.glob('*.csv'))
    assert len(files)==1, (folder.name,'expected one CSV')
    rows=list(csv.DictReader(files[0].open(encoding='utf-8-sig',newline='')))
    raw[folder.name]=rows
    signatures[folder.name]=Counter(tuple(sorted(r.items())) for r in rows)
    html[folder.name]={}
    for v in ('A','B'):
        p=EmailHTML();p.feed((folder/f'variant_{v}.html').read_text())
        html[folder.name][v]=p
    canonical[folder.name]=folder.name

duplicate='dach_de_plus_m_consistency26m'
keep='dach_de_plus_m_consistency26m_3'
assert signatures[duplicate]==signatures[keep], 'Duplicate assumption failed'
canonical[duplicate]=keep
issues.append({'campaign':duplicate,'severity':'excluded_duplicate','issue':'All 500 event records duplicate '+keep,'action':'Retained sanitized source copy; excluded from combined data.'})
names=[n for n in raw if canonical[n]==n]
campaign_ids={n:f'C{i:02d}' for i,n in enumerate(names,1)}
variant_map={};features=[];source_manifest=[]
for name in names:
    rows=raw[name];by_id=defaultdict(list)
    for r in rows:by_id[r['Email Campaign ID']].append(r)
    periods=sorted(by_id,key=lambda c:min(r['Sent Date (Your time zone)'] for r in by_id[c]))
    latest=periods[-1] if len(periods)==3 else None
    for cid, rr in by_id.items():
        subjects=set(norm(r['Subject']) for r in rr)
        matches=[v for v in ('A','B') if norm(html[name][v].title) in subjects]
        if name=='dach_de_free_business26_4':
            v='A' if cid=='439251478' else 'B'
            if v=='B':assert all(norm(r['Subject']).endswith(', heute endet dein Preisvorteil.') for r in rr)
        else:
            assert len(matches)==1,(name,cid,'ambiguous variant mapping');v=matches[0]
        stage=('rollout_fallback' if SCREEN[name][6]=='inconclusive_fallback' else 'rollout_winner') if cid==latest else 'test'
        variant_map[(name,cid)]=(v,stage)
    family='newsletter_CW12_2025' if 'CW12-25' in name else name
    for v in ('A','B'):
        p=html[name][v];sub=norm(p.title)
        if name=='dach_de_free_business26_4' and v=='B':sub='[RECIPIENT], heute endet dein Preisvorteil.'
        features.append({'campaign_id':campaign_ids[name],'campaign_name':name,'content_family':family,'variant':v,
          'subject_template':sub,'subject_template_characters':len(sub),'subject_has_question':int('?' in sub),
          'subject_has_number':int(bool(re.search(r'\d',sub))),'subject_personalized':int('[RECIPIENT]' in sub),
          'html_image_elements':p.images,'html_link_elements':p.links,'body_text_words':len(' '.join(p.parts).split()),
          'body_text_sha256':hashlib.sha256('\n'.join(p.parts).encode()).hexdigest(),
          'body_text_matches_other_variant':int(p.parts==html[name]['B' if v=='A' else 'A'].parts)})
    screen=SCREEN[name]
    if screen[4]=='unavailable':issues.append({'campaign':name,'severity':'legacy','issue':'Bot filtering unavailable in source report.','action':'Exclude from primary filtered analysis; retain for sensitivity analysis.'})
    if any('@' not in r['Recipient'] for r in rows):issues.append({'campaign':name,'severity':'identity_missing','issue':'Hidden recipients cannot be linked across sends.','action':'Use message-scoped unknown IDs, never merge all hidden contacts.'})
    if any(dt(r['Sent Date (Your time zone)'])+timedelta(days=7)>OBSERVATION_END for r in rows):
        issues.append({'campaign':name,'severity':'incomplete_window','issue':'Seven-day outcome window incomplete as of export date.','action':'All seven-day labels left blank; re-export after September 25, preferably September 26 or later.'})

feature_lookup={(f['campaign_name'],f['variant']):f for f in features}
event_rows=[];messages=defaultdict(list);event_keys=set();original_emails=set();personal_subjects=set()
for folder,rows in raw.items():
    name=canonical[folder];clean=[]
    for r in rows:
        scope=r['Hub ID']+'|';cid=r['Email Campaign ID'];message_key=scope+cid+'|'+r['Message ID']
        v,stage=variant_map[(name,cid)]
        addr=r['Recipient'].strip().lower();known=bool(EMAIL.fullmatch(addr))
        recipient=token('person',addr) if known else token('unknown_message',message_key)
        if known:original_emails.add(addr)
        if name=='dach_de_free_business26_4' and v=='B':personal_subjects.add(r['Subject'])
        internal=int(known and addr.rsplit('@',1)[-1] in {'provenexpert.com','bewertet.de'})
        e={'campaign_id':campaign_ids[name],'campaign_name':name,'source_campaign_id':cid,'variant':v,'phase':stage,
           'recipient_id':recipient,'recipient_linkable':int(known),'internal_domain_candidate':internal,
           'message_id':token('message',message_key),'event_id':token('event',scope+r['Event ID']),
           'event_type':r['Event Type'],'sent_at_local':r['Sent Date (Your time zone)'],
           'event_at_local':r['Event Created Date (Your time zone)'],
           'subject_template':feature_lookup[(name,v)]['subject_template'],
           'event_superseded':int(bool(r['Obsoleted by Event ID'])),
           'superseded_by_event_id':token('event',scope+r['Obsoleted by Event ID']) if r['Obsoleted by Event ID'] else '',
           'caused_by_event_id':token('event',scope+r['Caused by Event ID']) if r['Caused by Event ID'] else '',
           'link_id':token('link',r['Click URL']),
           'not_sent_reason':r['Not Sent Reason'] if re.fullmatch('[A-Z_]*',r['Not Sent Reason']) else 'REDACTED',
           'bot_filtering_reported':SCREEN[name][4]}
        clean.append(e)
        if folder==name:
            k=(r['Hub ID'],r['Event ID'])
            assert k not in event_keys,'unexpected duplicate event'
            event_keys.add(k);event_rows.append(e);messages[e['message_id']].append(e)
    write_csv(Path('pseudonymized_events')/folder/'events.csv',clean)
    source_manifest.append({'source_folder':folder,'canonical_campaign':name,'campaign_id':campaign_ids[name],
      'source_event_rows':len(rows),'included_in_combined':int(folder==name),'duplicate_of':name if folder!=name else '',
      'source_csv_sha256':hashlib.sha256(next((ROOT/folder).glob('*.csv')).read_bytes()).hexdigest(),
      'all_five_expected_files_present':1})

send_rows=[]
for mid,ee in messages.items():
    active=[e for e in ee if not e['event_superseded']]
    active_sends=[e for e in active if e['event_type']=='SENT']
    delivered=[e for e in active if e['event_type']=='DELIVERED']
    if not active_sends and not delivered:continue
    e=ee[0];sent=min(dt(x['sent_at_local']) for x in ee)
    cutoff=sent+timedelta(days=7);mature=cutoff<=OBSERVATION_END
    clk=sorted(dt(x['event_at_local']) for x in active if x['event_type']=='CLICK' and dt(x['event_at_local'])>=sent)
    opn=sorted(dt(x['event_at_local']) for x in active if x['event_type']=='OPEN' and dt(x['event_at_local'])>=sent)
    # Retain latest observed outcomes for reconciliation, even after the conservative analysis cutoff.
    clicks7=[t for t in clk if t<=cutoff]
    r={k:e[k] for k in ['campaign_id','campaign_name','source_campaign_id','variant','phase','recipient_id','recipient_linkable','internal_domain_candidate','message_id','subject_template','bot_filtering_reported']}
    r.update({'sent_at_local':sent.strftime('%Y-%m-%d %H:%M:%S'),'sent_event_active':int(bool(active_sends)),
      'delivered':int(bool(delivered)),'opened_observed':int(bool(opn)),'clicked_observed':int(bool(clk)),
      'click_events_observed':len(clk),'first_click_delay_hours':round((clk[0]-sent).total_seconds()/3600,4) if clk else '',
      'window_7d_complete':int(mature),'clicked_7d':int(bool(clicks7)) if mature and delivered else '',
      'analysis_cutoff_local':OBSERVATION_END.strftime('%Y-%m-%d %H:%M:%S'),
      'configured_p_A_given_test':0.5 if e['phase']=='test' else '',
      'assignment_mechanism_verified':0,
      'content_family':feature_lookup[(e['campaign_name'],e['variant'])]['content_family'],
      'eligible_primary_pilot':int(bool(delivered) and mature and e['phase']=='test' and e['bot_filtering_reported']=='on' and not e['internal_domain_candidate'])})
    send_rows.append(r)

# Prior features use only events dated before the send, not eventual earlier-email labels.
history=defaultdict(lambda:{'deliveries':[],'clicks':[],'click_delays':[]})
for mid,ee in messages.items():
    e=ee[0]
    if not e['recipient_linkable']:continue
    for typ,target in [('DELIVERED','deliveries'),('CLICK','clicks')]:
        times=[dt(x['event_at_local']) for x in ee if not x['event_superseded'] and x['event_type']==typ]
        if times:history[e['recipient_id']][target].append(min(times))
for h in history.values():
    h['deliveries'].sort();h['clicks'].sort()
for r in send_rows:
    if not r['recipient_linkable']:
        r.update({'prior_delivered_in_exports':'','prior_clicked_messages_in_exports':'','days_since_prior_click_in_exports':''})
    else:
        h=history[r['recipient_id']];t=dt(r['sent_at_local']);n=bisect.bisect_left(h['clicks'],t)
        r.update({'prior_delivered_in_exports':bisect.bisect_left(h['deliveries'],t),
          'prior_clicked_messages_in_exports':n,'days_since_prior_click_in_exports':round((t-h['clicks'][n-1]).total_seconds()/86400,4) if n else ''})
send_rows.sort(key=lambda r:(r['sent_at_local'],r['campaign_id'],r['message_id']))

audits=[];variants=[]
for name in names:
    ee=[e for e in event_rows if e['campaign_name']==name];ss=[r for r in send_rows if r['campaign_name']==name]
    s=SCREEN[name]
    active=lambda typ:len({e['message_id'] for e in ee if e['event_type']==typ and not e['event_superseded']})
    a={'campaign_id':campaign_ids[name],'campaign_name':name,'send_date':min(r['sent_at_local'] for r in ss)[:10],
      'events':len(ee),'messages_with_active_send_or_delivery':len(ss),'active_sent_events':active('SENT'),
      'delivered':active('DELIVERED'),'opened_messages':active('OPEN'),'clicked_messages':active('CLICK'),
      'dashboard_sent':s[0],'dashboard_delivered':s[1],'dashboard_opened':s[2],'dashboard_clicked':s[3],
      'delivery_delta':active('DELIVERED')-s[1],'open_delta':active('OPEN')-s[2],'click_delta':active('CLICK')-s[3],
      'hidden_delivered_messages':sum(r['delivered'] and not r['recipient_linkable'] for r in ss),
      'bot_filtering_reported':s[4],'test_duration_hours':s[5],'test_outcome':s[6],'rollout_variant':s[7],
      'test_split_A':0.25 if s[5] else 0.5,'test_split_B':0.25 if s[5] else 0.5,'rollout_share':0.5 if s[5] else 0,
      'winning_metric':'opens_per_delivery' if s[5] else 'not_specified_even_split',
      'mature_delivered_7d':sum(r['delivered'] and r['window_7d_complete'] for r in ss),
      'clicked_7d':sum(r['clicked_7d']==1 for r in ss) if any(r['delivered'] and r['window_7d_complete'] for r in ss) else '',
      'eligible_primary_pilot':sum(r['eligible_primary_pilot'] for r in ss)}
    audits.append(a)
    for metric in ['delivery','open','click']:
        if a[metric+'_delta']:
            issues.append({'campaign':name,'severity':'reconciliation','issue':metric+' counts differ from performance screenshot by '+str(a[metric+'_delta']),
              'action':'Keep exported outcomes, retain screenshot totals separately; do not fabricate missing events.'})
    if active('SENT')!=s[0]:issues.append({'campaign':name,'severity':'reconciliation','issue':'Active SENT count differs from screenshot by '+str(active('SENT')-s[0]),'action':'Retain discrepancy explicitly; use delivered population for click pilot.'})
    for (v,phase) in sorted({(r['variant'],r['phase']) for r in ss}):
        dd=[r for r in ss if r['variant']==v and r['phase']==phase and r['delivered']]
        n=len(dd);k=sum(r['clicked_observed'] for r in dd);phat=k/n if n else 0;z=1.96
        den=1+z*z/n if n else 1;center=(phat+z*z/(2*n))/den if n else 0
        half=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den if n else 0
        variants.append({'campaign_id':campaign_ids[name],'campaign_name':name,'variant':v,'phase':phase,
          'delivered':n,'clicked_observed':k,'click_rate_observed':round(phat,6),
          'wilson95_low_descriptive':round(center-half,6),'wilson95_high_descriptive':round(center+half,6),
          'mature_7d_delivered':sum(r['window_7d_complete'] for r in dd),
          'clicked_7d':sum(r['clicked_7d']==1 for r in dd) if any(r['window_7d_complete'] for r in dd) else ''})

write_csv(Path('combined_events.csv'),event_rows)
write_csv(Path('recipient_send_table.csv'),send_rows)
write_csv(Path('primary_pilot_7d.csv'),[r for r in send_rows if r['eligible_primary_pilot']])
write_csv(Path('campaign_audit.csv'),audits)
write_csv(Path('variant_summary.csv'),variants)
write_csv(Path('email_features.csv'),features)
write_csv(Path('source_manifest.csv'),source_manifest)
write_csv(Path('quality_issues.csv'),issues)

known_campaigns=defaultdict(set)
for r in send_rows:
    if r['recipient_linkable'] and r['delivered']:known_campaigns[r['recipient_id']].add(r['campaign_id'])
stats={'input_folders':len(raw),'distinct_campaigns':len(names),'input_event_rows':sum(len(x) for x in raw.values()),
 'unique_event_rows':len(event_rows),'duplicate_rows_removed':sum(len(x) for x in raw.values())-len(event_rows),
 'delivered_messages':sum(r['delivered'] for r in send_rows),
 'clicked_delivered_messages_observed':sum(r['delivered'] and r['clicked_observed'] for r in send_rows),
 'mature_delivered_messages':sum(r['delivered'] and r['window_7d_complete'] for r in send_rows),
 'clicked_7d':sum(r['clicked_7d']==1 for r in send_rows),
 'hidden_delivered_messages':sum(r['delivered'] and not r['recipient_linkable'] for r in send_rows),
 'linkable_recipients_with_delivery':len(known_campaigns),
 'linkable_recipients_in_multiple_campaigns':sum(len(v)>1 for v in known_campaigns.values()),
 'primary_pilot_rows':sum(r['eligible_primary_pilot'] for r in send_rows),
 'primary_pilot_clicks_7d':sum(r['eligible_primary_pilot'] and r['clicked_7d']==1 for r in send_rows),
 'primary_pilot_rows_with_prior_delivery':sum(r['eligible_primary_pilot'] and isinstance(r['prior_delivered_in_exports'],int) and r['prior_delivered_in_exports']>0 for r in send_rows),
 'internal_candidate_deliveries':sum(r['delivered'] and r['internal_domain_candidate'] for r in send_rows),
 'analysis_cutoff_local':str(OBSERVATION_END)}

# Privacy checks inspect the exact CSV payloads: whitelist schema, no source addresses,
# raw personal subjects, URLs or unmasked message/event/recipient identifiers exported.
for f in OUT.rglob('*.csv'):
    content=f.read_text()
    assert not EMAIL.search(content),('email in output',f.name)
    assert 'http://' not in content and 'https://' not in content,('url in output',f.name)
    for s in personal_subjects:
        if s not in {'Liebe/r Experte:in, heute endet dein Preisvorteil.'}:
            assert s not in content,('personalized subject in output',f.name)
assert len({r['message_id'] for r in send_rows})==len(send_rows)
assert all(r['clicked_7d']=='' for r in send_rows if not r['window_7d_complete'])
assert all(r['recipient_id'].startswith('unknown_message_') for r in send_rows if not r['recipient_linkable'])
assert all(a['delivery_delta']==0 and a['click_delta']==0 for a in audits)
assert len({token('person',e) for e in original_emails})==len(original_emails)
stats['validation']='passed: CSV privacy scan, unique send keys, no immature labels, delivery/click dashboard reconciliation'
(OUT/'summary.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
