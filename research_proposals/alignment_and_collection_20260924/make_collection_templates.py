"""Generate planned rows only. Blank measured values are deliberately unknown."""
from pathlib import Path
import csv
import json
from collections import Counter

OUT = Path(__file__).parent
CLASSES = ['normal','inBroken','outBroken','roll']
RPMS = [1000,2000,3000]
PLANNED = ['plan_id','stage','required','session_plan','split_fixed_speed',
           'fault_type_plan','rpm_setpoint_plan','environment_plan','distance_cm_plan',
           'azimuth_deg_plan','elevation_deg_plan','min_stable_seconds_plan',
           'target_contact_complete_packets','radar_cfg_policy','planned_order',
           'reference_policy','run_status']
MEASURED = ['recording_id','attempt_id','parent_failed_attempt','operator_id','session_date',
            'session_actual','independent_restart_id','bearing_entity_id','motor_id',
            'bearing_mount_id','contact_mount_id','radar_pose_id','environment_actual',
            'distance_cm_actual','distance_reference','azimuth_deg_actual','elevation_deg_actual',
            'rpm_setpoint_actual','rpm_feedback_mean','rpm_feedback_min','rpm_feedback_max',
            'rpm_evidence_type','rpm_log_path','load_setting','load_evidence',
            'ambient_temp_C','motor_temp_start_C','motor_temp_end_C','temperature_source',
            'warmup_minutes','elapsed_motor_running_minutes','contact_sensor_model',
            'contact_sample_rate_hz','contact_sample_rate_evidence','contact_range',
            'contact_units','contact_filter_setting','contact_axis_description',
            'serial_baud','serial_framing','contact_payload_format','contact_packet_samples',
            'radar_board_id','radar_cfg_path','radar_cfg_sha256','radar_tx_mask',
            'radar_rx_gain_dB','radar_fw_version','radar_sdk_dfp_version','capture_sw_version',
            'dca_fw_version','dca_packet_delay','dca_packet_loss_count',
            'radar_start_utc','radar_stop_utc','contact_start_utc','contact_stop_utc',
            'clock_source','clock_domain_id','stable_start_marker','stable_end_marker',
            'stable_marker_semantics','stable_duration_s','radar_path','contact_raw_path',
            'contact_timing_path','dca_log_path','cli_log_path','photo_path',
            'radar_bytes','contact_bytes','observed_raw_byte_rate','contact_packets_started',
            'contact_complete_packets','contact_invalid_records','contact_counter_breaks',
            'contact_crc_failures','radar_valid_chirp_fraction','adc_rail_fraction',
            'adc_bit_alignment_confirmed','range_peak_m','qc_status','qc_rule_version',
            'qc_reason','fault_label_source','physical_change_notes','raw_sha256_manifest_path']


def row(plan_id,stage,session,cls,rpm,env,distance,split,order,required='yes'):
    data = {k:'' for k in PLANNED+MEASURED}
    data.update(plan_id=plan_id,stage=stage,required=required,session_plan=session,
                split_fixed_speed=split,fault_type_plan=cls,rpm_setpoint_plan=rpm,
                environment_plan=env,distance_cm_plan=distance,
                azimuth_deg_plan=0,elevation_deg_plan=0,min_stable_seconds_plan=30,
                target_contact_complete_packets=20,radar_cfg_policy='one_pilot_accepted_cfg_for_all_main_rows',
                planned_order=order,reference_policy='training_only_no_test_normal_reference',
                run_status='PLANNED_NOT_COLLECTED')
    return data


def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=PLANNED+MEASURED); w.writeheader();w.writerows(rows)


core=[]
orders=[CLASSES,['inBroken','roll','normal','outBroken'],
        ['outBroken','normal','roll','inBroken'],['roll','outBroken','inBroken','normal']]
for si,session in enumerate('ABCD'):
    order=0
    for ci,cls in enumerate(orders[si]):
        rpms=RPMS[(si+ci)%3:]+RPMS[:(si+ci)%3]
        if (si+ci)%2:rpms=list(reversed(rpms))
        for rpm in rpms:
            order+=1
            core.append(row(f'CORE-{session}-{order:02d}','core',session,cls,rpm,'room',40,
                            'train' if session in 'AB' else ('validation' if session=='C' else 'sealed_test'),order))
write('collection_manifest_core48.csv',core)

extension=[]
for si,session in enumerate(['E','F']):
    order=0
    environments=['room','narrow'] if session=='E' else ['narrow','room']
    for env in environments:
        classes=orders[si] if env=='room' else list(reversed(orders[si]))
        for ci,cls in enumerate(classes):
            distances=([40,20,80] if (ci+si)%2==0 else [80,40,20]) if env=='room' else [40]
            for d in distances:
                order+=1
                extension.append(row(f'EXT-{session}-{order:02d}','geometry_environment',session,cls,
                                     2000,env,d,'sealed_external_test',order))
write('collection_manifest_extension32.csv',extension)

combined=[]
for si,session in enumerate(['E','F']):
    for ci,cls in enumerate(orders[si]):
        combined.append(row(f'COMB-{session}-{ci+1:02d}','optional_combined_shift',session,cls,
                            2000,'narrow',80,'sealed_external_test',ci+1,'optional'))
write('collection_manifest_optional_combined8.csv',combined)

keep=[]
for si,session in enumerate('ABCD'):
    for ri,rpm in enumerate(RPMS):
        split='contact_head_only_train' if session in 'AB' else ('contact_head_only_validation' if session=='C' else 'new_fault_sealed_test')
        keep.append(row(f'KEEP-{session}-{ri+1:02d}','optional_new_fault',session,'keep',rpm,
                        'room',40,split,ri+1,'optional'))
write('collection_manifest_optional_keep12.csv',keep)

off=[]
for session in 'ABCDEF':
    conditions=[('room',40)] if session in 'ABCD' else [('room',40),('room',20),('room',80),('narrow',40)]
    for env,d in conditions:
        for when in ['before','after']:
            r=row(f'OFF-{session}-{env}-{d}-{when}','background',session,'off',0,env,d,
                  'background_reserved',len(off)+1)
            r['min_stable_seconds_plan']=10
            r['target_contact_complete_packets']=''
            r['reference_policy']='P0_excluded_from_preprocess_P1_report_allowed_background'
            off.append(r)
write('collection_manifest_background24.csv',off)
write('collection_manifest_main80.csv',core+extension)

assert len(core)==48 and len(extension)==32 and len(combined)==8 and len(keep)==12 and len(off)==24
assert len({r['plan_id'] for r in core+extension+combined+keep+off})==124
assert all(not r[k] for r in core+extension+combined+keep+off for k in MEASURED)
assert Counter((r['session_plan'],r['fault_type_plan']) for r in core)==Counter({(s,c):3 for s in 'ABCD' for c in CLASSES})
out=dict(core48=len(core),recommended_extension32=len(extension),main80=len(core+extension),
         optional_combined8=len(combined),optional_keep12=len(keep),background24=len(off),
         measured_fields_blank=True, no_capture_or_training_performed=True,
         split_notice='For held-out-rpm experiments, override all local fitting masks: exclude target rpm from teacher adaptation, references, scaling, mapping, student training and validation selection.')
(OUT/'collection_template_verification.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
