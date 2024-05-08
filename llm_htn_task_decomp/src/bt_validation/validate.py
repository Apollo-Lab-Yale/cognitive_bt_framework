import xmlschema

def validate_bt(xml) :
  schema = xmlschema.XMLSchema('llm_htn_task_decomp/src/bt_validation/bt_schema.xsd')
  try:
      schema.validate(xml)
      #print("Valid")
      return("Valid")
  except xmlschema.validators.exceptions.XMLSchemaValidationError as e:
      #print(e)
      return(e)