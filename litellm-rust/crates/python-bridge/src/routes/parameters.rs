use std::collections::BTreeSet;

use litellm_core_utils::call_arguments::CallArguments;
use litellm_host_python::{from_py, lookup};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::{Map, Value};

pub(super) fn supplied_keys(kwargs: &Bound<'_, PyDict>) -> PyResult<BTreeSet<String>> {
    kwargs.keys().iter().map(|key| key.extract()).collect()
}

pub(super) fn provider_parameters(
    py: Python<'_>,
    prepared: &Bound<'_, PyDict>,
    bound: &Bound<'_, PyDict>,
    supplied: &BTreeSet<String>,
    inputs: &[&str],
    provider_owned: &[&str],
) -> PyResult<CallArguments> {
    let owned = py
        .import("litellm.types.utils")?
        .getattr("is_litellm_owned_kwarg")?;
    project_parameters(prepared, bound, supplied, inputs, provider_owned, &|name| {
        owned.call1((name,))?.extract()
    })
}

fn project_parameters(
    prepared: &Bound<'_, PyDict>,
    bound: &Bound<'_, PyDict>,
    supplied: &BTreeSet<String>,
    inputs: &[&str],
    provider_owned: &[&str],
    owned: &impl Fn(&str) -> PyResult<bool>,
) -> PyResult<CallArguments> {
    let accepts = |name: &str| -> PyResult<bool> {
        if inputs.contains(&name)
            || litellm_core_utils::params::is_control_param(name)
            || matches!(
                name,
                "model"
                    | "extra_body"
                    | "base_url"
                    | "default_headers"
                    | "api_version"
                    | "organization"
                    | "deployment_id"
                    | "headers"
                    | "provider_specific_header"
                    | "callbacks"
                    | "success_callback"
                    | "failure_callback"
            )
        {
            return Ok(false);
        }
        Ok(provider_owned.contains(&name) || !owned(name)?)
    };
    let names = bound
        .keys()
        .iter()
        .chain(prepared.keys().iter())
        .map(|key| key.extract::<String>())
        .collect::<PyResult<BTreeSet<_>>>()?;
    let fields = names
        .into_iter()
        .filter_map(|name| {
            let project = || -> PyResult<Option<(String, Value)>> {
                if !accepts(&name)? {
                    return Ok(None);
                }
                let Some(value) = lookup(prepared, bound.as_any(), &name)? else {
                    return Ok(None);
                };
                if value.is_none()
                    && !supplied.contains(&name)
                    && bound.get_item(&name)?.is_some_and(|value| value.is_none())
                {
                    return Ok(None);
                }
                Ok(Some((name.clone(), from_py(&value)?)))
            };
            project().transpose()
        })
        .collect::<PyResult<Map<String, Value>>>()?;
    let overrides = lookup(prepared, bound.as_any(), "extra_body")?
        .filter(|value| !value.is_none())
        .map(|value| {
            let mapping = value
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("extra_body must be an object"))?;
            mapping
                .iter()
                .filter_map(|(key, value)| {
                    let project = || -> PyResult<Option<(String, Value)>> {
                        let name = key.extract::<String>()?;
                        if !accepts(&name)? {
                            return Ok(None);
                        }
                        Ok(Some((name, from_py(&value)?)))
                    };
                    project().transpose()
                })
                .collect::<PyResult<Map<String, Value>>>()
        })
        .transpose()?;
    let arguments: CallArguments = fields
        .into_iter()
        .chain(overrides.map(|fields| ("extra_body".to_string(), Value::Object(fields))))
        .collect();
    arguments
        .resolve_body_overrides()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[cfg(test)]
mod tests {
    use litellm_host_python::to_py;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::null(json!(null))]
    #[case::false_value(json!(false))]
    #[case::zero(json!(0))]
    #[case::nested(json!({"mode": "new", "values": [true, null, {"nested": 7}]}))]
    fn unknown_fields_survive_projection(#[case] extension: Value) {
        Python::initialize();
        Python::attach(|py| {
            let source = json!({"future_provider_option": extension});
            let bound = to_py(py, &source).unwrap();
            let bound = bound.bind(py).cast::<PyDict>().unwrap();
            let parameters = project_parameters(
                &PyDict::new(py),
                bound,
                &supplied_keys(bound).unwrap(),
                &[],
                &[],
                &|_| Ok(false),
            )
            .unwrap();
            assert_eq!(serde_json::to_value(parameters).unwrap(), source);
        });
    }

    #[rstest]
    fn projection_resolves_prepared_values_and_overrides_without_serializing_controls() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
opaque = object()
bound = {'model': 'resolved', 'messages': [], 'temperature': 0.25,
         'future_default': None, 'future_null': None, 'future_cleared': True,
         'metadata': {'user_id': 'keep'}, 'callbacks': [opaque], 'api_key': opaque}
prepared = {'future_cleared': None, 'temperature': 0.5,
            'extra_body': {'temperature': 0.75, 'future_override': None,
                           'model': 'ignored', 'messages': ['ignored'],
                           'litellm_metadata': opaque, 'callbacks': [opaque], 'api_key': opaque}}
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let bound = locals.get_item("bound").unwrap().unwrap();
            let prepared = locals.get_item("prepared").unwrap().unwrap();
            let parameters = project_parameters(
                prepared.cast::<PyDict>().unwrap(),
                bound.cast::<PyDict>().unwrap(),
                &BTreeSet::from(["future_null".to_string()]),
                &["messages"],
                &["metadata"],
                &|name| Ok(name.starts_with("litellm_") || name == "metadata"),
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(parameters).unwrap(),
                json!({"temperature": 0.75, "future_null": null, "future_cleared": null,
                       "future_override": null, "metadata": {"user_id": "keep"}})
            );
            assert!(
                bound
                    .get_item("api_key")
                    .unwrap()
                    .is(locals.get_item("opaque").unwrap().unwrap())
            );
        });
    }

    #[rstest]
    #[case::boolean(json!(false))]
    #[case::number(json!(0))]
    #[case::array(json!([]))]
    #[case::string(json!(""))]
    fn invalid_overrides_are_terminal(#[case] overrides: Value) {
        Python::initialize();
        Python::attach(|py| {
            let bound = to_py(py, &json!({"extra_body": overrides})).unwrap();
            let error = project_parameters(
                &PyDict::new(py),
                bound.bind(py).cast::<PyDict>().unwrap(),
                &BTreeSet::new(),
                &[],
                &[],
                &|_| Ok(false),
            )
            .unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }
}
