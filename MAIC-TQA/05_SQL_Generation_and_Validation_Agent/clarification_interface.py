

def SELECT_AmbiguityClarification_interface(string, data):
    """
    Extract the value of 'column name' from 'SELECT-clarification' in data.
    If 'SELECT-clarification' is an empty dictionary or the key is missing, return 'no'.
    """
    follow_up = data.get('SELECT_clarification', {})
    if data['intent_ambiguity'] == True:
    #if follow_up and isinstance(follow_up, dict) and '列头' in follow_up:
        follow_answer = "我问的是" + follow_up['列名'] + "。"
        return follow_answer
    else:
        return '无'


def FROM_AmbiguityClarification_interface(detection, data):
    """
    Extract the correct value of the slot from 'FROM_clarification' in data.
    If the label does not exist in 'scope_ambiguity', return 'no'.
    """
    detection = list(detection)
    if detection in data['scope_ambiguity']:
        if detection[1] == 'city' and detection[2] != 'Correct':
            follow_up = [k for k, v in data['FROM_clarification'].items() if v == "城市"]
            follow_answer = f"正确的城市应该是{follow_up[0]}"
            return follow_answer
        elif detection[1] == 'district' and detection[2] != 'Correct':
            follow_up = [k for k, v in data['FROM_clarification'].items() if v == "区域"]
            follow_answer = f"正确的区域应该是{follow_up[0]}"
            return follow_answer
        elif detection[1] == 'year' and detection[2] != 'Correct':
            follow_up = [k for k, v in data['FROM_clarification'].items() if v == "年份"]
            follow_answer = f"我关注的年份是{follow_up[0]}"
            return follow_answer
        else:
            return '无'
    else:
        return '无'


def WHERE_AmbiguityClarification_interface(detection, data):
    detection = list(detection)
    if detection in data['condition_ambiguity']:
        ambiguity_name = detection[0]
        real_names = data['WHERE_clarification'].keys()
        real_name = ''
        for name in real_names:
            if set(ambiguity_name).issubset(set(name)):
                real_name = name
        if real_name != '':
            follow_answer = f"正确的{detection[1]}应该是{real_name}。"
            return follow_answer
        else:
            return '无'
    else:
        return '无'













