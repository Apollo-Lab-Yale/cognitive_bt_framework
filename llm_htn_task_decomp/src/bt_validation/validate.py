import xmlschema

def validate_bt(xml) :
  schema = xmlschema.XMLSchema('./bt_schema.xsd')
  try:
      schema.validate(xml)
      #print("Valid")
      return("Valid")
  except xmlschema.validators.exceptions.XMLSchemaValidationError as e:
      #print(e)
      return(e)