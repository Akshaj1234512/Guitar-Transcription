import torch
import torch.nn.functional as F

def masked_mse(student, teacher, mask, eps=1e-8):
    """
    student, teacher: (B, T, C) or (B, T, K) same shape
    mask:            (B, T, C) broadcastable to student
    """
    diff2 = (student - teacher)**2
    diff2 = diff2 * mask
    return diff2.sum() / (mask.sum() + eps)

def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)

############ High-resolution regression loss ############
def regress_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


# def consistency_loss(student_dict, teacher_dict, target_dict,
#                      tau=0.3, w_frame=1.0, w_onset=0.2, eps=1e-8):
#     """
#     Enforce: student(aug) ≈ teacher(clean)
#     Only where mask_roll is valid, and only enforce onset where teacher is confident.
#     """
#     mask = target_dict['mask_roll']  # (B,T,88) presumably

#     # teacher probs
#     t_frame = teacher_dict['frame_output'].detach()
#     t_onset = teacher_dict['reg_onset_output'].detach()

#     # student probs
#     s_frame = student_dict['frame_output']
#     s_onset = student_dict['reg_onset_output']

#     # frame consistency everywhere valid
#     loss_frame = masked_mse(s_frame, t_frame, mask, eps)

#     # onset consistency only where teacher believes note exists
#     note_gate = (t_frame > tau).float()
#     onset_mask = mask * note_gate
#     loss_onset = masked_mse(s_onset, t_onset, onset_mask, eps)

#     return w_frame * loss_frame + w_onset * loss_onset

def consistency_loss(student_dict, teacher_dict, target_dict,
                     tau=0.0, w_frame=1.0, w_onset=1.0, eps=1e-8):
    """
    IMPROVED: Uses Importance Weighting instead of Hard Gating.
    Focuses the Student on matching the Teacher's PEAKS.
    """
    mask = target_dict['mask_roll']

    # --- 1. Frame Consistency (Sustain) ---
    # We want consistency everywhere, but we can trust the teacher more 
    # when it is confident (either definitely ON or definitely OFF).
    t_frame = teacher_dict['frame_output'].detach()
    s_frame = student_dict['frame_output']
    
    loss_frame = masked_mse(s_frame, t_frame, mask, eps)

    # --- 2. Onset Consistency (The Fix) ---
    t_onset = teacher_dict['reg_onset_output'].detach()
    s_onset = student_dict['reg_onset_output']
    
    # PROBLEM WITH OLD CODE: 
    # note_gate = (t_frame > 0.3) <--- This filtered out the start of notes!
    
    # NEW STRATEGY: "Focus on the Attacks"
    # We create a weight map. We want the student to pay 
    # EXTRA attention where the Teacher detects an onset.
    
    # If Teacher says onset=0.9, weight is 1 + (5 * 0.9) = 5.5
    # If Teacher says onset=0.0, weight is 1.0
    onset_importance = 1.0 + (5.0 * t_onset)
    
    # Calculate Squared Error
    squared_diff = (s_onset - t_onset) ** 2
    
    # Apply Mask AND Importance Weights
    # This forces the student to prioritize the 'clicks' even if distortion hides them
    weighted_diff = squared_diff * onset_importance * mask
    
    loss_onset = torch.sum(weighted_diff) / (torch.sum(mask) + eps)

    return w_frame * loss_frame + w_onset * loss_onset


def regress_onset_offset_frame_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    total_loss = onset_loss + offset_loss + frame_loss
    return total_loss


def regress_pedal_bce(model, output_dict, target_dict):
    """High-resolution piano pedal regression loss, including pedal onset 
    regression, pedal offset regression and pedal frame-wise classification losses.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['reg_pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['reg_pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss

############ Google's onsets and frames system loss ############
def google_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano note loss. Only used for comparison.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def google_pedal_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano pedal loss. Only used for comparison.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss


def get_loss_func(loss_type):
    if loss_type == 'regress_onset_offset_frame_velocity_bce':
        return regress_onset_offset_frame_velocity_bce

    elif loss_type == 'regress_pedal_bce':
        return regress_pedal_bce

    elif loss_type == 'google_onset_offset_frame_velocity_bce':
        return google_onset_offset_frame_velocity_bce

    elif loss_type == 'google_pedal_bce':
        return google_pedal_bce

    elif loss_type == "regress_onset_offset_frame_bce":
        return regress_onset_offset_frame_bce


    else:
        raise Exception('Incorrect loss_type!')