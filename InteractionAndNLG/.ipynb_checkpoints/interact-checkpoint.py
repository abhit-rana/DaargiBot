def give_greetings_reply():
    return "Hey! How can I help you?"

def give_bye_reply():
    return "Bye!"

def extract_symptoms(info_dict):
    

def interact():
    While(True):
        user_input = input("Please enter the text:")
        info_dict = get_info_dict(user_input)
        if(info_dict['user_intent']=='greetings'):
            print(give_greetings_reply())
        elif(info_dict['user_intent']=='self_report'):
            extract_symptoms(info_dict)
        elif(info_dict['user_intent']=='good_bye'):
            print(give_bye_reply)