data_df['Assoc Rules Score'] = 0 
for assoc_rule_idx in assoc_rules.index: 
    antecedents = assoc_rules.loc[assoc_rule_idx, 'antecedents']
    consequent = assoc_rules.loc[assoc_rule_idx, 'consequents']
    support = assoc_rules.loc[assoc_rule_idx, 'support']
    cond = True
    col_list = (list(antecedents))
    for col_name in col_list:
        cond = cond & (data_df[col_name])        
    fis_true_list = data_df[cond].index 
    col_list = (list(consequent))
    for col_name in col_list:
        cond = cond & (data_df[col_name])
    assoc_rule_true_list = data_df[cond].index 
        
    rule_exceptions = set(fis_true_list) - set(assoc_rule_true_list) 
    data_df.loc[rule_exceptions, 'Assoc Rules Score'] += support 
